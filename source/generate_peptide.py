import os
from typing import Iterator, Tuple, Dict, List


def read_fasta(fasta_path: str) -> Iterator[Tuple[str, str]]:
    """
    读取 FASTA 文件
    返回: (header, sequence)
    """
    header = None
    seq_chunks = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)

        if header is not None:
            yield header, "".join(seq_chunks)


def parse_uniprot_header(header: str) -> Tuple[str, str]:
    """
    尝试解析 UniProt FASTA header
    例如:
    sp|P31946|1433B_HUMAN 14-3-3 protein beta/alpha OS=Homo sapiens ...
    
    返回:
    protein_id, protein_name
    """
    protein_id = ""
    protein_name = ""

    parts = header.split("|")
    if len(parts) >= 3:
        protein_id = parts[1].strip()
        rest = parts[2].strip()
        protein_name = rest.split(" OS=")[0].strip()
    else:
        protein_id = header.split()[0]
        protein_name = header

    return protein_id, protein_name


def clean_sequence(seq: str) -> str:
    """
    清洗序列:
    - 转大写
    - 去掉空格
    - 只保留标准氨基酸字母
    """
    seq = seq.upper().replace(" ", "")
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join([aa for aa in seq if aa in valid_aas])


def sliding_peptides(seq: str, k: int) -> Iterator[Tuple[int, int, str]]:
    """
    对单条蛋白序列做长度为 k 的滑动窗口切分
    返回:
    start(1-based), end(1-based), peptide
    """
    n = len(seq)
    if n < k:
        return

    for i in range(n - k + 1):
        peptide = seq[i:i + k]
        start = i + 1
        end = i + k
        yield start, end, peptide


def generate_peptides_from_fasta(
    fasta_path: str,
    output_dir: str,
    lengths: List[int] = [8, 9, 10, 11]
) -> None:
    """
    从 FASTA 生成不同长度的肽段，并分别保存到不同文件
    """
    os.makedirs(output_dir, exist_ok=True)

    output_files: Dict[int, str] = {
        k: os.path.join(output_dir, f"peptides_{k}mer.tsv")
        for k in lengths
    }

    writers = {}
    handles = {}

    try:
        # 打开输出文件并写表头
        for k, path in output_files.items():
            f = open(path, "w", encoding="utf-8")
            handles[k] = f
            f.write("protein_id\tprotein_name\tstart\tend\tpeptide\n")
            writers[k] = f

        protein_count = 0

        for header, raw_seq in read_fasta(fasta_path):
            protein_id, protein_name = parse_uniprot_header(header)
            seq = clean_sequence(raw_seq)

            if not seq:
                continue

            protein_count += 1

            for k in lengths:
                if len(seq) < k:
                    continue

                for start, end, peptide in sliding_peptides(seq, k):
                    writers[k].write(
                        f"{protein_id}\t{protein_name}\t{start}\t{end}\t{peptide}\n"
                    )

        print(f"处理完成，共处理蛋白数量: {protein_count}")
        for k, path in output_files.items():
            print(f"{k}-mer 输出文件: {path}")

    finally:
        for f in handles.values():
            f.close()


if __name__ == "__main__":
    fasta_path = "/home/mclab/mjp/unifyimmun/data/uniprotkb_proteome_UP000005640_2026_03_28.fasta"   # 改成你的 FASTA 路径
    output_dir = "/home/mclab/mjp/unifyimmun/data/peptides"        # 输出目录

    generate_peptides_from_fasta(
        fasta_path=fasta_path,
        output_dir=output_dir,
        lengths=[8, 9, 10, 11]
    )