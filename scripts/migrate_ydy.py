#!/usr/bin/env python3
"""ydy8989.github.io (Jekyll/beautiful-jekyll) → Astro 이관.

- 카테고리 하위폴더의 글을 평평하게 옮긴다 (파일명 = permalink 슬러그).
- permalink `/:year-:month-:day-:title/` 를 그대로 재현하도록 slug 를 심는다.
- 외부 이미지를 내려받아 로컬로 교체(옵션), 로컬 상대경로도 표준화.
- Disqus 스레드 보존을 위해 원래 절대 URL을 frontmatter 에 기록.
"""
import re, sys, json, hashlib, pathlib, urllib.request, urllib.parse

REPO = pathlib.Path("/Users/dodo/Documents/code/2026/ydy8989.github.io")
SRC = REPO / "_posts"
DST = REPO / "src/content/blog"
IMGDIR = REPO / "public/assets/posts"
SITE = "https://ydy8989.github.io"

# 전량 이관: 필터 비움.
ONLY: set[str] = set()
DOWNLOAD_IMAGES = "--no-img" not in sys.argv

FNAME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.+)$")

# 날짜 접두사가 없어 Jekyll 이 무시하던 초안. 라이브에 URL 이 없으므로
# (보존할 스레드도 없음) git 최초 커밋일 기준으로 슬러그를 새로 부여한다.
OVERRIDE = {
    "2021roberta": ("2021", "07", "21", "roberta"),
}


def split_fm(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    return (m.group(1), m.group(2)) if m else ("", text)


def fm_get(fm, key):
    m = re.search(rf"^{key}\s*:\s*(.*)$", fm, re.M)
    return m.group(1).strip().strip('"').strip("'") if m else None


def fm_list(fm, key):
    raw = fm_get(fm, key)
    if not raw:
        return []
    return [t.strip().strip('"').strip("'") for t in raw.strip("[]").split(",") if t.strip()]


def q(s):
    return '"' + (s or "").replace('\\', '').replace('"', "'") + '"'


def download(url, slug):
    ext = ".png"
    for e in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".PNG", ".JPG"):
        if e in url:
            ext = e.lower().replace(".jpeg", ".jpg")
            break
    name = f"{slug}-{hashlib.md5(url.encode()).hexdigest()[:8]}{ext}"
    dest = IMGDIR / name
    if not dest.exists():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                dest.write_bytes(r.read())
        except Exception as e:
            print(f"    ! 이미지 실패 {url[:70]}: {e}", file=sys.stderr)
            return None
    return f"/assets/posts/{name}"


IMG_MD = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\s*(?:\"[^\"]*\")?\))")


def rewrite_images(body, slug, stats):
    def repl(m):
        pre, url, post = m.group(1), m.group(2), m.group(3)
        if url.startswith("http"):
            stats["ext"] += 1
            if DOWNLOAD_IMAGES:
                local = download(url, slug)
                if local:
                    stats["ext_ok"] += 1
                    return pre + local + post
            return pre + url + post
        # 로컬 상대경로: ../../assets/img/... 또는 /assets/img/... → /assets/img/...
        stats["loc"] += 1
        norm = re.sub(r"^(\.\./)+", "/", url)
        if not norm.startswith("/"):
            norm = "/" + norm
        return pre + norm + post

    return IMG_MD.sub(repl, body)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    IMGDIR.mkdir(parents=True, exist_ok=True)
    report = []

    for f in sorted(SRC.rglob("*.md")):
        stem = f.stem
        m = FNAME_RE.match(stem)
        if m:
            y, mo, d, title_slug = m.groups()
        elif stem in OVERRIDE:
            y, mo, d, title_slug = OVERRIDE[stem]
        else:
            print(f"  건너뜀(날짜형식 아님): {f.name}")
            continue
        slug = f"{y}-{mo}-{d}-{title_slug}"
        if ONLY and slug not in ONLY:
            continue

        fm, body = split_fm(f.read_text(encoding="utf-8"))
        title = fm_get(fm, "title") or title_slug
        tags = fm_list(fm, "tags")
        cats = fm_list(fm, "categories") or ([f.parent.name] if f.parent != SRC else [])

        stats = {"ext": 0, "ext_ok": 0, "loc": 0}
        body = rewrite_images(body, slug, stats).strip()

        permalink = f"/{slug}/"
        disqus_url = f"{SITE}{permalink}"

        out = f"""---
title: {q(title)}
description: {q(fm_get(fm, "description") or fm_get(fm, "subtitle") or title)}
pubDate: {y}-{mo}-{d}
slug: {q(slug)}
categories: [{", ".join(q(c) for c in cats)}]
tags: [{", ".join(q(t) for t in tags)}]
disqusId: {q(disqus_url)}
---

{body}
"""
        (DST / f"{slug}.md").write_text(out, encoding="utf-8")
        report.append({"slug": slug, "permalink": permalink, "title": title,
                       "cats": cats, **stats})

    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\n총 {len(report)}편")


if __name__ == "__main__":
    main()
