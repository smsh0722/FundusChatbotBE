# pdf_to_text.py (요지)
import fitz, sys, pathlib

def pdf_to_text(p):
    doc = fitz.open(p)
    parts=[]
    for i in range(len(doc)):
        page = doc[i]
        # 1) 텍스트 블록 기반 추출 (헤딩/단락 구분에 유리)
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b:(b[1], b[0]))  # (y, x) 정렬
        for _,_,_,_,txt,_,_ in blocks:
            if txt.strip(): parts.append(txt.strip())
    return "\n\n".join(parts)

if __name__ == "__main__":
    inp = pathlib.Path(sys.argv[1])
    out = inp.with_suffix(".txt")
    out.write_text(pdf_to_text(str(inp)), encoding="utf-8")
    print("OK:", out)
