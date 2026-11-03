"""Build the retrieval corpus for RAG (data/rag/corpus.jsonl).

The fine-tune learned the answer style but not the facts, so instead of trying
to bake facts into 3B weights we look them up at answer time. Sources are
general public references, not picked per eval question:

  attack  MITRE ATT&CK Enterprise (pinned release): every technique, group,
          software and mitigation, with its description and detection notes
  kev     CISA Known Exploited Vulnerabilities catalog: one entry per CVE
  owasp   OWASP Cheat Sheet Series (web app security how-tos)
  wiki    Wikipedia articles in a fixed list of security categories
  train   the assistant answers from data/seed_scaled.jsonl

data/eval.jsonl is never read here. Run scripts/decontam.py --corpus afterwards
to confirm no passage copies an eval question.

Run: python scripts/build_corpus.py                 # all sources
     python scripts/build_corpus.py --sources attack kev train
"""
import argparse
import json
import os
import re
import urllib.parse
import urllib.request

ATTACK_URL = ("https://raw.githubusercontent.com/mitre/cti/ATT%26CK-v17.1/"
              "enterprise-attack/enterprise-attack.json")
KEV_URL = ("https://raw.githubusercontent.com/cisagov/kev-data/develop/"
           "known_exploited_vulnerabilities.json")
OWASP_RAW = "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/"
WIKI_API = "https://en.wikipedia.org/w/api.php"
# Broad categories chosen for coverage of the four domains, not per question.
WIKI_CATEGORIES = [
    "Computer security exploits", "Computer network security", "Cryptographic attacks",
    "Computer access control", "Authentication methods", "Side-channel attacks",
    "Email authentication", "Network protocols", "Computer forensics",
    "Malware", "Cyberattacks", "Computer security software",
]
TRAIN = "data/seed_scaled.jsonl"
OUT = "data/rag/corpus.jsonl"
CACHE = "data/rag/cache"
CHUNK_WORDS = 220      # passage length; title is repeated on every chunk


def fetch(url, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        if "/cheatsheets/" not in url:       # one line per source, not per file
            print(f"  downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def clean(text):
    text = re.sub(r"\(Citation:[^)]*\)", "", text or "")       # ATT&CK citations
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)       # markdown links
    text = re.sub(r"<[^>]+>", " ", text)                       # html tags
    return re.sub(r"[ \t]+", " ", text).strip()


def chunks(source, title, body):
    words = clean(body).split()
    for i in range(0, max(len(words), 1), CHUNK_WORDS):
        part = " ".join(words[i:i + CHUNK_WORDS])
        if part:
            yield {"source": source, "title": title, "text": part}


def attack():
    objs = json.load(open(fetch(ATTACK_URL, "enterprise-attack-v17.1.json")))["objects"]
    kinds = {"attack-pattern": "Technique", "intrusion-set": "Group",
             "malware": "Software", "tool": "Software", "course-of-action": "Mitigation"}
    for o in objs:
        if o.get("type") not in kinds or o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        ext = next((r["external_id"] for r in o.get("external_references", [])
                    if r.get("source_name") == "mitre-attack"), "")
        title = f"ATT&CK {ext} {o['name']} ({kinds[o['type']]})"
        body = o.get("description", "")
        if o.get("x_mitre_detection"):
            body += "\n\nDetection: " + o["x_mitre_detection"]
        if o.get("aliases"):
            body = "Aliases: " + ", ".join(o["aliases"]) + ". " + body
        yield from chunks("attack", title, body)


def kev():
    for v in json.load(open(fetch(KEV_URL, "kev.json")))["vulnerabilities"]:
        title = f"{v['cveID']} {v.get('vulnerabilityName', '')}"
        body = (f"{v.get('vendorProject', '')} {v.get('product', '')}. "
                f"{v.get('shortDescription', '')} Added to CISA KEV {v.get('dateAdded', '')}. "
                f"Known ransomware use: {v.get('knownRansomwareCampaignUse', 'Unknown')}. "
                f"Required action: {v.get('requiredAction', '')}")
        yield from chunks("kev", title, body)


def owasp():
    index = open(fetch(OWASP_RAW + "Index.md", "owasp-index.md")).read()
    for path in sorted(set(re.findall(r"cheatsheets/[A-Za-z0-9_\-]+\.md", index))):
        name = os.path.basename(path)[:-3]
        text = open(fetch(OWASP_RAW + path, f"owasp-{name}.md"), encoding="utf-8").read()
        text = re.sub(r"```.*?```", " ", text, flags=re.S)        # drop code blocks
        # split on headings so each passage stays on one sub-topic
        for sec in re.split(r"\n#{1,3} ", text):
            head, _, body = sec.partition("\n")
            yield from chunks("owasp", f"OWASP {name.replace('_', ' ')}: {head.strip('# ')}", body)


def wiki():
    """Plain-text Wikipedia articles from a fixed set of security categories."""
    def api(**params):
        params.update(format="json", formatversion="2")
        url = WIKI_API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "cybersec_agent-rag/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)

    cache = os.path.join(CACHE, "wiki.jsonl")
    if not os.path.exists(cache):
        titles = set()
        for cat in WIKI_CATEGORIES:
            cont = {}
            while True:
                d = api(action="query", list="categorymembers", cmtitle=f"Category:{cat}",
                        cmnamespace=0, cmlimit=500, **cont)
                titles.update(m["title"] for m in d["query"]["categorymembers"])
                if "continue" not in d:
                    break
                cont = {"cmcontinue": d["continue"]["cmcontinue"]}
        print(f"  wiki: {len(titles)} articles from {len(WIKI_CATEGORIES)} categories")
        os.makedirs(CACHE, exist_ok=True)
        with open(cache, "w") as f:
            titles = sorted(titles)
            for i in range(0, len(titles), 20):           # extracts API: 20 per call
                d = api(action="query", prop="extracts", explaintext=1, exlimit=20,
                        titles="|".join(titles[i:i + 20]))
                for pg in d["query"]["pages"]:
                    if pg.get("extract"):
                        f.write(json.dumps({"title": pg["title"], "text": pg["extract"]}) + "\n")
    for line in open(cache):
        a = json.loads(line)
        body = re.split(r"\n== (See also|References|External links|Further reading) ==", a["text"])[0]
        for sec in re.split(r"\n==+ ", body):
            head, _, rest = sec.partition("\n")
            title = a["title"] if not rest else f"{a['title']}: {head.strip('= ')}"
            yield from chunks("wiki", title, rest or head)


def train():
    for line in open(TRAIN):
        if not line.strip():
            continue
        r = json.loads(line)
        q, a = r["messages"][1]["content"], r["messages"][2]["content"]
        yield from chunks("train", q[:160], a)


SOURCES = {"attack": attack, "kev": kev, "owasp": owasp, "wiki": wiki, "train": train}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w") as f:
        for s in args.sources:
            k = 0
            try:
                for p in SOURCES[s]():
                    p["id"] = f"{s}-{k}"
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    k += 1
            except OSError as e:          # network blocked: keep the other sources
                print(f"  {s}: skipped after {k} passages ({e})")
            print(f"  {s:7s} {k:6d} passages")
            n += k
    print(f"Wrote {n} passages -> {OUT}")


if __name__ == "__main__":
    main()
