"""Tistory 공개 글 → Chirpy _posts 일회성 이전 스크립트.

usage: python tools/migrate_tistory.py <urls.txt>
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

ROOT = Path(__file__).resolve().parent.parent
POSTS = ROOT / "_posts"
IMG = ROOT / "assets" / "img" / "posts"
UA = {"User-Agent": "Mozilla/5.0"}

CATEGORY = {
    "DEVELOP_NOTE/ML": "ML & Modeling",
    "DEVELOP_NOTE/Statistics": "ML & Modeling",
    "DEVELOP_NOTE/Math": "ML & Modeling",
    "DEVELOP_NOTE/그 외": "Dev Notes",
    "DEVELOP_NOTE/Python": "Dev Notes",
    "알쓸신잡": "Dev Notes",
    "DEVELOP_NOTE/Linux": "MLOps & Infra",
    "DEVELOP_NOTE/MLOps": "MLOps & Infra",
    "DEVELOP_NOTE/Docker": "MLOps & Infra",
    "Error_Log": "Troubleshooting",
    "회고": "Retrospective",
    "DEVELOP_NOTE/LLM": "AI Engineering",
    "PAPER_REVIEW": "Paper Review",
    "PAPER_REVIEW/Medical_AI": "Paper Review",
}
# Tistory 하이라이터가 잘못 붙인 언어는 버린다
KNOWN_LANG = {"python", "bash", "shell", "sql", "json", "yaml", "javascript", "java", "html", "css", "cpp", "c", "go", "dockerfile", "plaintext"}


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


class Conv(MarkdownConverter):
    def convert_pre(self, el, text, parent_tags):
        lang = next((c for c in (el.get("class") or []) if c in KNOWN_LANG), "")
        code = el.get_text()
        return f"\n\n```{lang}\n{code.rstrip()}\n```\n\n"


def migrate(url):
    entry_id = url.rstrip("/").rsplit("/", 1)[-1]
    soup = BeautifulSoup(fetch(url).decode("utf-8", "replace"), "html.parser")
    title = soup.find("meta", property="og:title")["content"]
    published = soup.find("meta", property="article:published_time")["content"]  # 2022-10-11T15:58:15+09:00
    m = re.search(r'"categoryLabel":"([^"]*)"', str(soup))
    raw_cat = m.group(1) if m else ""
    tags = []
    for a in soup.select("a[rel=tag]"):
        tags += [t.strip().lower() for t in re.split(r"[#,]", a.get_text()) if t.strip()]

    body = soup.select_one("div.contents_style")
    # 이미지: 로컬로 내려받고 경로 교체 (카카오 CDN 의존 제거)
    for n, img in enumerate(body.find_all("img"), 1):
        src = img.get("src") or ""
        if "kakaocdn" not in src and "daumcdn" not in src:
            continue
        ext = Path(urllib.parse.urlparse(src).path).suffix or ".png"
        dest = IMG / entry_id / f"{n}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(fetch(src))
            img.replace_with(soup.new_tag("img", src=f"/assets/img/posts/{entry_id}/{n}{ext}", alt=img.get("alt") or ""))
        except Exception as e:  # 실패 이미지는 원본 링크 유지, 로그로 남김
            print(f"  ! image fail {entry_id}#{n}: {e}", file=sys.stderr)
    for fig in body.find_all("figure"):
        fig.unwrap()

    md = Conv(heading_style="ATX", bullets="-").convert(str(body))
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if "{{" in md or "{%" in md:
        md = "{% raw %}\n" + md + "\n{% endraw %}"

    date = published.replace("T", " ").replace("+09:00", " +0900")
    fm = {
        "title": title,
        "date": date,
        "categories": [CATEGORY.get(raw_cat, "Dev Notes")],
        "tags": sorted(set(tags)),
        "tistory_url": url,
    }
    front = "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in fm.items()) + "\n---\n\n"
    out = POSTS / f"{published[:10]}-{entry_id}.md"
    out.write_text(front + md + "\n", encoding="utf-8")
    return out.name, raw_cat


if __name__ == "__main__":
    urls = [u.strip() for u in Path(sys.argv[1]).read_text().splitlines() if u.strip()]
    POSTS.mkdir(exist_ok=True)
    ok = 0
    for u in urls:
        try:
            name, cat = migrate(u)
            ok += 1
            print(f"ok  {name}  [{cat}]")
        except Exception as e:
            print(f"ERR {u}: {e}", file=sys.stderr)
        time.sleep(0.3)
    print(f"done {ok}/{len(urls)}")
