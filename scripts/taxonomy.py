"""Topic taxonomy driving dataset generation — the diversity engine.

Naive "generate 1000 examples" produces near-duplicates. Instead we generate
a few distinct examples per *topic*, across many topics, to force coverage and
variety. Domains are weighted toward the target balance (education 35 / blue 25
/ threat_intel 25 / offensive 15).

THREAT_INTEL is deliberately a list of *specific, real* entities (CVEs, ATT&CK
IDs, frameworks) — we ground fact-heavy generation to real items rather than
letting the model invent CVE numbers or technique names. These rows should
still get a fact-check pass before training.
"""

SYSTEM = ("You are a cybersecurity assistant for authorized security testing, "
          "CTFs, and education. Explain clearly and include detection and "
          "mitigation where relevant.")

# Target domain balance (fractions) — curate.py reports against this.
TARGET = {"education": 0.35, "blue_team": 0.25, "threat_intel": 0.25, "offensive": 0.15}

EDUCATION = [
    "how HTTPS certificate chains and certificate authorities establish trust",
    "what a Content Security Policy is and which attacks it mitigates",
    "the difference between a firewall, WAF, and a proxy",
    "how OAuth 2.0 authorization code flow works and its security properties",
    "what session fixation is and how to prevent it",
    "the difference between authentication factors (knowledge, possession, inherence)",
    "how DNSSEC works and what it protects against",
    "what a rainbow table is and why salting defeats it",
    "how public key infrastructure (PKI) manages keys and revocation",
    "what clickjacking is and how X-Frame-Options / frame-ancestors help",
    "the difference between confidentiality and integrity controls",
    "how a VPN protects traffic and what it does and doesn't hide",
    "what perfect forward secrecy is and why it matters",
    "how password managers improve security posture",
    "what a security misconfiguration is with common examples",
    "the difference between a bug bounty and a penetration test",
    "how rate limiting and throttling defend against abuse",
    "what defense in depth means with a layered example",
    "how certificate pinning works and its tradeoffs",
    "the difference between encryption in transit and at rest",
    "what an air-gapped network is and its limitations",
    "how single sign-on (SSO) works and its security implications",
    "what the difference is between a threat, vulnerability, and risk",
    "how sandboxing isolates untrusted code",
    "what a security operations center (SOC) does day to day",
    "how threat modeling (e.g., STRIDE) is performed",
    "the difference between red team, blue team, and purple team",
    "what data loss prevention (DLP) is and how it works",
    "how role-based vs attribute-based access control differ",
    "what a bastion host / jump box is and why it's used",
    "how HMAC provides message authentication",
    "what a nonce is and why reuse is dangerous",
    "the difference between symmetric block and stream ciphers",
    "how salting and peppering differ for password storage",
    "what the same-origin policy is and why it matters",
    "how mutual TLS (mTLS) authentication works",
    "what a supply-chain SBOM is and why it helps",
    "the difference between IDS signatures and anomaly detection",
    "what network segmentation and micro-segmentation achieve",
    "how a reverse proxy differs from a forward proxy for security",
]

BLUE_TEAM = [
    "Sigma rule to detect wmic.exe spawning a process on a remote host",
    "Sigma rule to detect rundll32.exe executing with suspicious exported functions",
    "Sigma rule to detect creation of a new Windows service via sc.exe",
    "Sigma rule to detect bitsadmin.exe being used to download a file",
    "Sigma rule to detect registry run-key persistence being created",
    "Sigma rule to detect ntdsutil or vssadmin access to NTDS.dit",
    "Sigma rule to detect suspicious wevtutil clear-log commands",
    "Sigma rule to detect PsExec-style service creation and named pipes",
    "YARA rule to flag files containing embedded PowerShell download cradles",
    "YARA rule to flag documents with suspicious VBA auto-exec macros",
    "triage steps for a suspicious DLL loaded from a user-writable directory",
    "triage steps when EDR flags credential access on a domain controller",
    "analyzing a proxy log showing beaconing at regular intervals to one host",
    "analyzing Windows Event 4624/4625 patterns indicating password spraying",
    "incident response for a confirmed ransomware detonation in progress",
    "incident response steps for a compromised service account",
    "how to scope lateral movement after finding one compromised host",
    "building a detection for token impersonation / access token theft",
    "interpreting Sysmon Event ID 1 for suspicious parent-child chains",
    "triage of an alert for outbound traffic to a newly registered domain",
    "detecting DLL search-order hijacking on Windows",
    "hunting for pass-the-hash activity in authentication logs",
    "detecting golden ticket / forged Kerberos ticket usage",
    "responding to a phishing email reported by a user, step by step",
    "detecting data staging before exfiltration (large archive creation)",
    "building a detection for cleartext credentials in command lines",
    "triage of anomalous cloud IAM activity (new access keys, policy changes)",
    "how to validate whether an IDS alert is a true or false positive",
    "detecting living-off-the-land use of mshta.exe or regsvr32.exe",
    "log analysis of a web server showing repeated 401s then a 200 to /admin",
]

OFFENSIVE = [
    "authorized CTF methodology for exploiting server-side template injection",
    "authorized methodology for testing CORS misconfiguration in a web app",
    "authorized CTF approach to a heap exploitation challenge with tcache",
    "authorized methodology for testing OAuth/OIDC flows for security flaws",
    "authorized CTF approach to a ROP-chain challenge with NX enabled",
    "authorized methodology for testing subdomain takeover",
    "authorized approach to enumerating and abusing misconfigured S3 buckets in scope",
    "authorized CTF methodology for a padding-oracle attack",
    "authorized methodology for testing HTTP request smuggling",
    "authorized CTF approach to a Windows kernel privilege escalation lab",
    "authorized methodology for testing prototype pollution in a Node.js app",
    "authorized approach to abusing weak sudo rules for privilege escalation in a lab",
    "authorized CTF methodology for a ret2libc challenge",
    "authorized methodology for testing LDAP injection",
    "authorized approach to Kerberoasting in an assessment lab AD environment",
    "authorized methodology for testing NoSQL injection",
    "authorized CTF approach to a shellcode-writing exploitation challenge",
    "authorized methodology for testing WebSocket security in a web app",
]

# GROUNDED: specific real CVEs / ATT&CK techniques / frameworks. The generator
# is told to summarize the given real entity, not invent one.
THREAT_INTEL = [
    # CVEs (well-documented)
    "CVE-2014-6271 (Shellshock) Bash environment-variable RCE",
    "CVE-2018-13379 Fortinet FortiOS SSL VPN path traversal / credential disclosure",
    "CVE-2019-19781 Citrix ADC/NetScaler path traversal to RCE",
    "CVE-2020-0601 (CurveBall) Windows CryptoAPI ECC certificate validation flaw",
    "CVE-2022-1388 F5 BIG-IP iControl REST authentication bypass RCE",
    "CVE-2023-34362 MOVEit Transfer SQL injection (Cl0p mass exploitation)",
    "CVE-2023-3519 Citrix NetScaler ADC/Gateway unauthenticated RCE",
    "CVE-2022-30190 (Follina) Microsoft MSDT RCE via Office documents",
    "CVE-2021-40444 MSHTML RCE via malicious Office documents",
    "CVE-2020-5902 F5 BIG-IP TMUI RCE",
    # ATT&CK techniques (verify ID->name before training)
    "MITRE ATT&CK T1547 Boot or Logon Autostart Execution (registry run keys)",
    "MITRE ATT&CK T1021 Remote Services (RDP, SMB, SSH) for lateral movement",
    "MITRE ATT&CK T1027 Obfuscated Files or Information",
    "MITRE ATT&CK T1105 Ingress Tool Transfer",
    "MITRE ATT&CK T1112 Modify Registry",
    "MITRE ATT&CK T1567 Exfiltration Over Web Service",
    "MITRE ATT&CK T1136 Create Account (persistence)",
    "MITRE ATT&CK T1082 System Information Discovery",
    "MITRE ATT&CK T1573 Encrypted Channel for C2",
    "MITRE ATT&CK T1204 User Execution",
    "MITRE ATT&CK T1548 Abuse Elevation Control Mechanism (UAC bypass, sudo)",
    "MITRE ATT&CK T1218 System Binary Proxy Execution (LOLBins)",
    # frameworks / concepts (threat-intel analysis)
    "the MITRE ATT&CK framework's tactics vs techniques structure",
    "STIX and TAXII for sharing structured threat intelligence",
    "the difference between strategic, operational, and tactical threat intelligence",
    "what a threat actor 'campaign' and 'intrusion set' mean in CTI",
    "the Emotet malware family and its historical impact",
    "the Mirai IoT botnet and how it spread",
    "what MITRE D3FEND is and how it complements ATT&CK",
    "the concept of dwell time and mean time to detect (MTTD)",
]

DOMAINS = {
    "education": EDUCATION,
    "blue_team": BLUE_TEAM,
    "offensive": OFFENSIVE,
    "threat_intel": THREAT_INTEL,
}

if __name__ == "__main__":
    for d, topics in DOMAINS.items():
        print(f"{d:13s} {len(topics):3d} topics")
    total = sum(len(t) for t in DOMAINS.values())
    print(f"{'TOTAL':13s} {total:3d} topics")
