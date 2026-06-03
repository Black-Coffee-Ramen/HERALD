import whois
import dns.resolver

domain = "trzo-suites.wixstudio.com"
print("WHOIS:")
try:
    w = whois.whois(domain)
    print(w)
except Exception as e:
    print(e)

print("DNS A:")
try:
    answers = dns.resolver.resolve(domain, "A")
    for a in answers:
        print(a)
except Exception as e:
    print(e)

