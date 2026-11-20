import h5py
import numpy as np
import pandas as pd
import os
import re
from pyfaidx import Fasta
import cyvcf2  # replaces manual gzip/open VCF parsing


# --- CONFIGURATION ---
#VCF_DIR_PATH = "/project/dbnrrc/resources/zeroshot_TRJ_SNPs/by_chr_snpEff/"
VCF_DIR_PATH = "/project/dbnrrc/resources/zeroshot/vcf/"

# SHAP and Gene Data Paths
SHAP_H5_PATH = "/project/dbnrrc/jared/research/DeepCRE-CUDA/model_reimplemented/deepCRE_reimplemented/src/deepCRE/results/shap/oryzaAnther_deepcre_interpret_250925_213855.h5"
SHAP_META_CSV = "/project/dbnrrc/jared/research/DeepCRE-CUDA/model_reimplemented/deepCRE_reimplemented/src/deepCRE/results/shap/oryzaAnther_deepcre_interpret_250925_213856_shap_meta.csv"
GENE_COORDS_CSV = "../ory_files/inputCoords_shap_zssv60.csv"

# Reference FASTA
REFERENCE_FASTA = "/project/dbnrrc/jared/research/PlantCaduceus/src/ory_files/Oryza_sativa.IRGSP-1.0.dna.toplevelv60.fa"

# Output Directory
OUTPUT_DIR = "../ory_files/outputs/shap_direct_vcf/ANTHER/"
TISSUE_NAME = "Anther"

# Constants
INPUT_LENGTH  = 3020
BASES         = "ACGT"
INTRAGENIC    = 500   # bases extracted inward from TSS/TTS (into the gene body)
EXTRAGENIC    = 1000  # bases extracted outward from TSS/TTS (flanking region)
CENTRAL_PAD   = 20    # fixed N-padding between TSS and TTS windows

# Reverse complement tables
RC_MAP        = str.maketrans("ACGT", "TGCA")
RC_MAP_SINGLE = {"A": "T", "T": "A", "C": "G", "G": "C"}


def reverse_complement(seq):
    return seq.upper().translate(RC_MAP)[::-1]


def find_start_end(gene_start, gene_end, strand):
    """
    Mirrors DeepCRE's find_start_end logic from extract_sequences.py exactly.
    """
    import math
    gene_length = abs(gene_end - gene_start)
    if gene_length < 2 * INTRAGENIC:
        longer_intra  = math.ceil(gene_length / 2)
        shorter_intra = math.floor(gene_length / 2)
        additional_padding = 2 * INTRAGENIC - gene_length
    else:
        longer_intra = shorter_intra = INTRAGENIC
        additional_padding = 0

    if strand == "+":
        prom_start = gene_start - EXTRAGENIC
        prom_end   = gene_start + longer_intra
        term_start = gene_end   - shorter_intra
        term_end   = gene_end   + EXTRAGENIC
    else:
        prom_start = gene_end   - longer_intra
        prom_end   = gene_end   + EXTRAGENIC
        term_start = gene_start - EXTRAGENIC
        term_end   = gene_start + shorter_intra

    return prom_start, prom_end, term_start, term_end, additional_padding


def extract_gene_sequence(fasta, chrom, gene_start, gene_end, strand):
    """
    Reconstructs the 3020 bp DeepCRE input using the same logic as
    extract_sequences.py in the deepCRE_reimplemented repository.
    """
    prom_start, prom_end, term_start, term_end, additional_padding = \
        find_start_end(gene_start, gene_end, strand)

    total_pad = CENTRAL_PAD + additional_padding
    PAD = "N" * total_pad

    prom_seq = str(fasta[chrom][prom_start : prom_end]).upper()
    term_seq = str(fasta[chrom][term_start : term_end]).upper()

    if strand == "+":
        seq = prom_seq + PAD + term_seq
    else:
        seq = reverse_complement(prom_seq) + PAD + reverse_complement(term_seq)

    return seq  # always INPUT_LENGTH (3020) bp


def verify_ref_base(fasta, chrom, gene_start, gene_end, strand, input_idx, ref_base):
    """
    Confirms that the base at input_idx in the DeepCRE input window
    matches the REF allele from the VCF.
    """
    seq = extract_gene_sequence(fasta, chrom, gene_start, gene_end, strand)
    fasta_base = seq[input_idx].upper()
    expected_base = ref_base if strand == "+" else RC_MAP_SINGLE.get(ref_base, ref_base)
    return fasta_base == expected_base, fasta_base


def diagnose_ref_mismatch(fasta, chrom, gene_start, gene_end, strand,
                           pos, input_idx, ref_base, fasta_base):
    """
    Prints a detailed diagnostic when a REF base mismatch is detected.
    """
    CONTEXT = 10
    seq = extract_gene_sequence(fasta, chrom, gene_start, gene_end, strand)
    ctx_start = max(0, input_idx - CONTEXT)
    ctx_end   = min(len(seq), input_idx + CONTEXT + 1)
    left  = seq[ctx_start:input_idx]
    mid   = seq[input_idx]
    right = seq[input_idx + 1:ctx_end]
    marked = f"{left}[{mid}]{right}"

    print(f"    Gene coords : {chrom}:{gene_start}-{gene_end} ({strand})")
    print(f"    VCF pos     : {pos}  ->  input_idx={input_idx}")
    print(f"    VCF REF     : {ref_base}  |  FASTA base: {fasta_base}")
    print(f"    FASTA context (input coords {ctx_start}-{ctx_end}): {marked}")
    window_start = gene_start - 1000
    window_end   = gene_end   + 1000
    print(f"    Raw genomic window: {chrom}:{window_start}-{window_end} "
          f"({'revcomp' if strand == '-' else 'forward'})")


def load_shap_data(h5_path, meta_path):
    """Loads SHAP H5 data and maps Gene IDs to their H5 index."""
    print("Loading SHAP H5 file...")
    with h5py.File(h5_path, "r") as f:
        hypo_scores = f["hypothetical_contrib_scores"][:]

    print("Loading SHAP Metadata...")
    meta_df = pd.read_csv(meta_path)
    gene_id_to_idx = {gid: i for i, gid in enumerate(meta_df["gene_ids"])}

    return hypo_scores, gene_id_to_idx


# ---------------------------------------------------------------------------
# ANN parsing — gene-aware, multi-annotation aware
# ---------------------------------------------------------------------------

#def parse_ann_effects(ann_field, gene_id=None):
#    """
#    Parse all ANN entries from a SnpEff ANN field value (already extracted
#    from the INFO string, i.e. everything after "ANN=").
#
#    ANN format (pipe-delimited per entry, comma-delimited between entries):
#        ALT|effect|impact|gene_name|gene_id|feature_type|feature_id|...
#
#    Parameters
#    ----------
#    ann_field : str or None
#        The raw ANN field value (e.g. from cyvcf2's variant.INFO["ANN"]).
#    gene_id : str or None
#        If provided, only ANN entries whose gene_id field (index 4) matches
#        this value are kept. This prevents cross-gene annotation bleed when
#        a variant is in the window of multiple genes.
#        If None, all entries are returned (original behaviour).
#
#    Returns
#    -------
#    str or None
#        Semicolon-joined unique effects for the matching annotations, e.g.
#        "upstream_gene_variant;synonymous_variant".
#        Returns None if no matching annotations are found.
#    """
#    if not ann_field:
#        return None
#
#    effects = []
#    for entry in ann_field.split(","):
#        fields = entry.split("|")
#        if len(fields) < 5:
#            continue
#
#        ann_gene_id   = fields[4].strip()   # e.g. "Os01g0100100"
#        ann_effect    = fields[1].strip()   # e.g. "upstream_gene_variant"
#
#        # If a gene_id filter is given, skip entries that don't match
#        if gene_id is not None and ann_gene_id != gene_id:
#            continue
#
#        if ann_effect:
#            effects.append(ann_effect)
#
#    if not effects:
#        return None
#
#    # Deduplicate while preserving order
#    seen = set()
#    unique_effects = []
#    for e in effects:
#        if e not in seen:
#            seen.add(e)
#            unique_effects.append(e)
#
#    return ";".join(unique_effects)
def parse_ann_effects(ann_field, gene_id=None):
    """
    Parse ANN entries from a SnpEff ANN field value and return the effect
    from the FIRST transcript that matches the given gene_id.

    ANN format (pipe-delimited per entry, comma-delimited between entries):
        ALT|effect|impact|gene_name|gene_id|feature_type|...

    Parameters
    ----------
    ann_field : str or None
        Raw ANN field value from cyvcf2's variant.INFO["ANN"].
    gene_id : str or None
        If provided, only the first ANN entry whose gene_id field (index 4)
        matches is used.  Prevents cross-gene annotation bleed.
        If None, the first entry is returned regardless of gene.

    Returns
    -------
    str or None
        Effect string from the first matching transcript, e.g.
        "upstream_gene_variant", or None if no match found.
    """
    # Check if received an ANN string - when no ANN tag return early
    if not ann_field:
        return None

    # Split all ANN entries into a generator of field lists
    # A generator is used so next() can stop immediately on the first match
    # without processing every entry — direct lookup rather than a full loop
    entries = (entry.split("|") for entry in ann_field.split(","))

    # next() scans entries and stops at the first one that:
    #   (1) has enough fields to be valid (at least 5)
    #   (2) belongs to the gene we are currently processing (index 4 = gene_id)
    # use the first matching transcript's effect only
    # Note: AF is handled separately in parse_vcf_file, not here
    match = next(
        (fields for fields in entries
         if len(fields) >= 5
         and (gene_id is None or fields[4].strip() == gene_id)),
        None     # default if no match is found
    )

    if match is None:
        return None

    # SnpEff uses "|" - index 0 - ALT allele; index 1 - effect; index 2 - impact; index 3 - gene name; index 4 - gene_id; etc.
    return match[1].strip() or None     # index 1 is the effect from the first matching transcript



# ---------------------------------------------------------------------------
# VCF parsing with cyvcf2
# ---------------------------------------------------------------------------

def parse_vcf_file(vcf_path):
    """
    Parses a VCF (plain or gzipped) into a Pandas DataFrame using cyvcf2.

    cyvcf2 handles .vcf and .vcf.gz natively, provides typed INFO field
    access, and is significantly faster than line-by-line Python parsing.

    ANN effects are NOT extracted here because the correct filtering
    requires the gene_id being processed in the main loop.  Instead the
    raw ANN string is preserved so it can be filtered per-gene later.

    Multi-allelic sites are split: one row per ALT allele, with the
    corresponding per-allele AF value.
    """
    if not os.path.exists(vcf_path):
        print(f"File not found: {vcf_path}")
        return pd.DataFrame()

    data = []
    vcf = cyvcf2.VCF(vcf_path)

    for variant in vcf:
        chrom = variant.CHROM
        pos   = variant.POS          # 1-based, same as VCF POS
        ref   = variant.REF

        # INFO fields — cyvcf2 returns typed values (float, int, str, None)
        af_raw          = variant.INFO.get("AF")
        maf_raw         = variant.INFO.get("MAF")
        plant_cad_raw   = variant.INFO.get("plantCAD_zero_shot")
        ann_raw         = variant.INFO.get("ANN")   # full ANN string, filtered per-gene later

        # AF may be a single float or a list for multi-allelic sites
        if af_raw is None:
            afs = []
        elif isinstance(af_raw, (list, tuple)):
            afs = list(af_raw)
        else:
            afs = [af_raw]

        alts = variant.ALT  # list of ALT alleles

        # Pad AF list to match ALT count
        if len(afs) < len(alts):
            afs.extend([np.nan] * (len(alts) - len(afs)))

        for i, alt_allele in enumerate(alts):
            current_af = float(afs[i]) if (afs[i] is not None and not (
                isinstance(afs[i], float) and np.isnan(afs[i]))) else 0.0
            is_alt_major = current_af > 0.5

            data.append({
                "CHROM":              chrom,
                "POS":                int(pos),
                "REF":                ref,
                "ALT":                alt_allele,
                "AF":                 current_af,
                "MAF":                float(maf_raw) if maf_raw is not None else np.nan,
                "plantCAD_zero_shot": float(plant_cad_raw) if plant_cad_raw is not None else np.nan,
                "ANN_raw":            ann_raw,   # full ANN string; filtered per-gene in main loop
                "IS_ALT_MAJOR":       is_alt_major,
            })

    vcf.close()
    return pd.DataFrame(data)


def genomic_to_input_index(gene_start, gene_end, snp_pos, strand):
    """
    Maps a 1-based genomic VCF position to a 0-based index in the DeepCRE
    input sequence (0 to INPUT_LENGTH-1).
    Returns None if the position falls outside both input windows.
    """
    prom_start, prom_end, term_start, term_end, additional_padding = \
        find_start_end(gene_start, gene_end, strand)

    total_pad  = CENTRAL_PAD + additional_padding
    prom_len   = prom_end - prom_start
    tts_offset = prom_len + total_pad

    pos = snp_pos - 1   # convert to 0-based

    if strand == "+":
        if prom_start <= pos < prom_end:
            return pos - prom_start
        if term_start <= pos < term_end:
            return (pos - term_start) + tts_offset
    else:
        if prom_start <= pos < prom_end:
            return (prom_end - 1) - pos
        if term_start <= pos < term_end:
            return (term_end - 1) - pos + tts_offset

    return None


def debug_check_genomic_to_input_index(gene_start=10000, gene_end=20000, verbose=False):
    """
    Bug-checks for genomic_to_input_index().
    Covers: boundary correctness, strand symmetry,
    index uniqueness, gap body exclusion, and
    out-of-window rejection (both strands).
    """
    PASS = "PASS"
    FAIL = "FAIL"
    results = {}

    def check(name, condition):
        status = PASS if condition else FAIL
        results[name] = status
        if verbose:
            print(f"  [{status}] {name}")

    if verbose:
        print("\n=== genomic_to_input_index sanity checks ===\n")

    prom_start_p, prom_end_p, term_start_p, term_end_p, _ = find_start_end(gene_start, gene_end, "+")
    prom_start_m, prom_end_m, term_start_m, term_end_m, _ = find_start_end(gene_start, gene_end, "-")
    prom_len = prom_end_p - prom_start_p

    plus_tss_positions  = [p + 1 for p in range(prom_start_p, prom_end_p)]
    minus_tss_positions = [p + 1 for p in range(prom_start_m, prom_end_m)]
    plus_tts_positions  = [p + 1 for p in range(term_start_p, term_end_p)]
    minus_tts_positions = [p + 1 for p in range(term_start_m, term_end_m)]

    if verbose:
        print("── 1. TSS window ──")

    plus_tss  = [genomic_to_input_index(gene_start, gene_end, p, "+") for p in plus_tss_positions]
    minus_tss = [genomic_to_input_index(gene_start, gene_end, p, "-") for p in minus_tss_positions]

    if verbose:
        print(f"  TSS (+) sample [0,1,2]: {plus_tss[0]}, {plus_tss[1]}, {plus_tss[2]}")
        print(f"  TSS (-) sample [0,1,2]: {minus_tss[0]}, {minus_tss[1]}, {minus_tss[2]}")

    check("TSS (+) first index == 0",    plus_tss[0]   == 0)
    check("TSS (+) last index == 1499",  plus_tss[-1]  == prom_len - 1)
    check("TSS (-) first index == 1499", minus_tss[0]  == prom_len - 1)
    check("TSS (-) last index == 0",     minus_tss[-1] == 0)
    check("TSS (+) no None values", None not in plus_tss)
    check("TSS (-) no None values", None not in minus_tss)
    check("TSS (+) all unique", len(set(plus_tss)) == prom_len)
    check("TSS (-) all unique", len(set(minus_tss)) == prom_len)
    tss_sym_ok = all(plus_tss[i] == minus_tss[prom_len - 1 - i] for i in range(prom_len))
    check("TSS strand symmetry (plus[i] == minus[prom_len-1-i])", tss_sym_ok)

    if verbose:
        print("\n── 2. TTS window ──")

    plus_tts  = [genomic_to_input_index(gene_start, gene_end, p, "+") for p in plus_tts_positions]
    minus_tts = [genomic_to_input_index(gene_start, gene_end, p, "-") for p in minus_tts_positions]

    tts_first = prom_len + CENTRAL_PAD

    if verbose:
        print(f"  TTS (+) sample [0,1,2]: {plus_tts[0]}, {plus_tts[1]}, {plus_tts[2]}")
        print(f"  TTS (-) sample [0,1,2]: {minus_tts[0]}, {minus_tts[1]}, {minus_tts[2]}")

    check("TTS (+) first index == tts_first",       plus_tts[0]   == tts_first)
    check("TTS (+) last index == INPUT_LENGTH-1",   plus_tts[-1]  == INPUT_LENGTH - 1)
    check("TTS (-) first index == INPUT_LENGTH-1",  minus_tts[0]  == INPUT_LENGTH - 1)
    check("TTS (-) last index == tts_first",        minus_tts[-1] == tts_first)
    check("TTS (+) no None values", None not in plus_tts)
    check("TTS (-) no None values", None not in minus_tts)
    check("TTS (+) all unique", len(set(plus_tts)) == len(plus_tts_positions))
    check("TTS (-) all unique", len(set(minus_tts)) == len(minus_tts_positions))
    tts_len = len(plus_tts_positions)
    tts_sym_ok = all(plus_tts[i] == minus_tts[tts_len - 1 - i] for i in range(tts_len))
    check("TTS strand symmetry (plus[i] == minus[tts_len-1-i])", tts_sym_ok)

    if verbose:
        print("\n── 3. Gene body gap ──")

    gap_0based = list(range(prom_end_p, term_start_p))
    if gap_0based:
        gap_positions = [p + 1 for p in gap_0based]
        plus_gap  = [genomic_to_input_index(gene_start, gene_end, p, "+") for p in gap_positions]
        minus_gap = [genomic_to_input_index(gene_start, gene_end, p, "-") for p in gap_positions]
        check("Gap (+) all None", all(v is None for v in plus_gap))
        check("Gap (-) all None", all(v is None for v in minus_gap))
    else:
        check("Gap (+) all None (short gene, no gap)", True)
        check("Gap (-) all None (short gene, no gap)", True)

    if verbose:
        print("\n── 4. Out-of-window boundaries ──")

    out_positions = [
        prom_start_p,
        term_end_p + 1,
    ]
    if prom_end_p < term_start_p:
        out_positions += [
            prom_end_p + 1,
            term_start_p,
        ]
    for strand in ("+", "-"):
        out_results = [genomic_to_input_index(gene_start, gene_end, p, strand) for p in out_positions]
        check(f"Out-of-window ({strand}) all None",
              all(v is None for v in out_results))

    n_pass = sum(1 for v in results.values() if v == PASS)
    n_fail = sum(1 for v in results.values() if v == FAIL)

    if verbose:
        print(f"\n=== {n_pass}/{n_pass + n_fail} checks passed ===\n")

    if n_fail:
        print(f"\n[DEBUG] gene {gene_start}-{gene_end}: {n_fail}/{n_pass + n_fail} checks FAILED:")
        for name, status in results.items():
            if status == FAIL:
                print(f"  - {name}")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load SHAP Data
    shap_hypo, gene_idx_map = load_shap_data(SHAP_H5_PATH, SHAP_META_CSV)

    # 2. Load Gene Coordinates
    print("Loading Gene Coordinates...")
    gene_coords = pd.read_csv(GENE_COORDS_CSV)

    gene_map = {}
    for _, row in gene_coords.iterrows():
        chrom_clean = str(row["chrom"]).replace("Chr", "")
        gene_map[row["gene_id"]] = {
            "chrom":  chrom_clean,
            "start":  int(row["start"]),
            "end":    int(row["end"]),
            "strand": row.get("strand", "+")
        }

    # 3. Load reference FASTA once
    print("Loading reference FASTA...")
    fasta = Fasta(REFERENCE_FASTA)

    # 4. Main Loop
    for chrom_num in range(1, 13):
        chrom_str = f"Chr{chrom_num}"
        vcf_filename = f"{chrom_str}_snps.snpEff.filtered.vcf"
        vcf_path = os.path.join(VCF_DIR_PATH, vcf_filename)

        print(f"\n--- Processing {vcf_filename} ---")

        df_vars = parse_vcf_file(vcf_path)

        if df_vars.empty:
            print(f"Skipping {chrom_str} (No data found).")
            continue

        df_vars['chrom_num'] = df_vars['CHROM'].str.replace("Chr", "")
        df_vars = df_vars.sort_values("POS")
        var_positions = df_vars["POS"].values

        current_genes = [g for g in gene_idx_map.keys()
                         if g in gene_map and gene_map[g]["chrom"] == str(chrom_num)]

        annotated_results = []
        ref_mismatch_count = 0
        print(f"Mapping {len(current_genes)} genes to {len(df_vars)} variants...")

        for gene_id in current_genes:
            gene = gene_map[gene_id]

            debug_check_genomic_to_input_index(gene["start"], gene["end"])

            search_start = gene["start"] - 1000
            search_end   = gene["end"]   + 1000

            idx_start = np.searchsorted(var_positions, search_start, side='left')
            idx_end   = np.searchsorted(var_positions, search_end,   side='right')

            gene_vars = df_vars.iloc[idx_start:idx_end]

            if gene_vars.empty:
                continue

            for _, variant in gene_vars.iterrows():
                pos = variant["POS"]

                in_tss = gene["start"] - 1000 <= pos < gene["start"] + 500
                in_tts = gene["end"]   - 500  <= pos < gene["end"]   + 1000

                if not (in_tss or in_tts):
                    continue

                input_idx = genomic_to_input_index(
                    gene["start"], gene["end"], pos, gene["strand"])

                if input_idx is None or not (0 <= input_idx < INPUT_LENGTH):
                    continue

                ref = variant["REF"]
                alt = variant["ALT"]

                if ref not in BASES or alt not in BASES:
                    continue

                ref_match, fasta_base = verify_ref_base(
                    fasta, str(chrom_num),
                    gene["start"], gene["end"],
                    gene["strand"], input_idx, ref
                )
                if not ref_match:
                    ref_mismatch_count += 1
                    print(f"  [REF MISMATCH] {gene_id} pos={pos} "
                          f"input_idx={input_idx} strand={gene['strand']} "
                          f"VCF_REF={ref} FASTA={fasta_base}")
                    diagnose_ref_mismatch(
                        fasta, str(chrom_num),
                        gene["start"], gene["end"], gene["strand"],
                        pos, input_idx, ref, fasta_base
                    )
                    continue

                try:
                    h5_idx = gene_idx_map[gene_id]

                    if gene["strand"] == "-":
                        ref_lookup = RC_MAP_SINGLE.get(ref, ref)
                        alt_lookup = RC_MAP_SINGLE.get(alt, alt)
                    else:
                        ref_lookup = ref
                        alt_lookup = alt

                    ref_score = shap_hypo[h5_idx, input_idx, BASES.index(ref_lookup)]
                    alt_score = shap_hypo[h5_idx, input_idx, BASES.index(alt_lookup)]

                    if variant["IS_ALT_MAJOR"]:
                        major_score = alt_score
                        minor_score = ref_score
                    else:
                        major_score = ref_score
                        minor_score = alt_score

                    # ---------------------------------------------------------
                    # ANN: filter by current gene_id to get only relevant
                    # effects, and collect ALL matching effects (not just first)
                    # ---------------------------------------------------------
                    ann_effect = parse_ann_effects(variant["ANN_raw"], gene_id=gene_id)

                    annotated_results.append({
                        "Chr":                          variant["CHROM"],
                        "Pos":                          pos,
                        "GeneID":                       gene_id,
                        "Ref":                          ref,
                        "Alt":                          alt,
                        f"SHAP_Ref_{TISSUE_NAME}":      ref_score,
                        f"SHAP_Alt_{TISSUE_NAME}":      alt_score,
                        f"SHAP_Major_{TISSUE_NAME}":    major_score,
                        f"SHAP_Minor_{TISSUE_NAME}":    minor_score,
                        "AF":                           variant["AF"],
                        "MAF":                          variant["MAF"],
                        "plantCAD_zero_shot":           variant["plantCAD_zero_shot"],
                        "ANN_effect":                   ann_effect,
                    })

                except Exception as e:
                    continue

        if ref_mismatch_count > 0:
            print(f"  [WARNING] {ref_mismatch_count} REF mismatches skipped for {chrom_str}")

        if annotated_results:
            out_df = pd.DataFrame(annotated_results)
            save_path = f"{OUTPUT_DIR}{chrom_str}_SHAP_fullVCF_Annotated_06-01-26.csv"
            out_df.to_csv(save_path, index=False)
            print(f"Saved results to {save_path}")
        else:
            print(f"No overlaps found for {chrom_str}")


if __name__ == "__main__":
    main()
