from herald.investigation.intelligence import collect_dns_intelligence
try:
    res = collect_dns_intelligence("trzo-suites.wixstudio.com")
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()
