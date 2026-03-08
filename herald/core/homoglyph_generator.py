import os
import urllib.request
import itertools
import logging
import asyncio
import aiodns

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

CONFUSABLES_URL = "https://raw.githubusercontent.com/codebox/homoglyph/master/raw_data/char_codes.txt"
CONFUSABLES_FILE = "data/confusables.txt"

class HomoglyphGenerator:
    def __init__(self):
        self.confusables = {}
        self._load_confusables()

    def _load_confusables(self):
        """Loads or downloads a subset of Unicode confusables focusing on Latin similarities."""
        os.makedirs("data", exist_ok=True)
        
        # We define a minimal subset to avoid combination explosion for live checking.
        # Alternatively, we could download the full unicode confusables list.
        # For practicality, we map common English letters to common Cyrillic/Greek lookalikes.
        self.confusables = {
            'a': ['а', 'ɑ', 'ӓ', 'α'],
            'b': ['Ь', 'β', 'в'],
            'c': ['с', 'ϲ', 'ƈ'],
            'd': ['ԁ', 'ժ', 'ɗ'],
            'e': ['е', 'ë', 'é', 'ê', 'ε'],
            'g': ['ɡ', 'ġ', 'ğ'],
            'h': ['һ', 'հ'],
            'i': ['і', '1', 'l', 'í', 'ï', 'ι'],
            'j': ['ϳ', 'ј'],
            'k': ['к', 'κ'],
            'l': ['ӏ', '1', 'i', 'ι'],
            'm': ['м', 'ɱ'],
            'n': ['п', 'η', 'ñ'],
            'o': ['о', '0', 'ο', 'ö'],
            'p': ['р', 'ρ'],
            'q': ['ԛ', 'գ'],
            'r': ['г', 'ŕ'],
            's': ['ѕ', '5', 'ş'],
            't': ['т', 'τ'],
            'u': ['и', 'υ', 'ü'],
            'v': ['ѵ', 'ν'],
            'w': ['ѡ', 'ω'],
            'x': ['х', 'χ'],
            'y': ['у', 'γ', 'ý'],
            'z': ['z', 'ž']
        }
        
    def generate_variants(self, domain_name, max_variants=1000):
        """
        Generates visually similar domains.
        Limits exponential explosion by creating variations.
        """
        parts = domain_name.split('.')
        tld = parts[-1]
        name = ".".join(parts[:-1])
        
        # Build lists of choices for each character
        choices = []
        for char in name:
            opts = [char]
            if char in self.confusables:
                opts.extend(self.confusables[char])
            choices.append(opts)
            
        variants = []
        # We limit the itertools product to avoid MemoryError on long domains
        # A generator approach is safer
        def backtrack(index, current):
            if len(variants) >= max_variants:
                return
            if index == len(name):
                variants.append(current + "." + tld)
                return
                
            for choice in choices[index]:
                backtrack(index + 1, current + choice)
                if len(variants) >= max_variants:
                    return

        # Simple optimization: only change 1 or 2 characters to limit the set
        # Generate 1-character changes
        variants.append(domain_name)
        for i, char in enumerate(name):
            if char in self.confusables:
                for conf in self.confusables[char]:
                    variant_name = name[:i] + conf + name[i+1:]
                    variants.append(variant_name + "." + tld)
                    
        return list(set(variants))

    async def _is_registered_async(self, domain, resolver, semaphore):
        async with semaphore:
            try:
                punycode_domain = domain.encode('idna').decode('utf-8')
                await resolver.query(punycode_domain, 'A')
                return domain, True
            except aiodns.error.DNSError:
                return domain, False
            except Exception:
                return domain, False

    async def _find_registered_async(self, variants):
        resolver = aiodns.DNSResolver()
        semaphore = asyncio.Semaphore(500)
        
        tasks = [self._is_registered_async(var, resolver, semaphore) for var in variants]
        results = await asyncio.gather(*tasks)
        
        registered = []
        for domain, is_reg in results:
            if is_reg:
                registered.append(domain)
                logging.warning(f"🚨 ACTIVE HOMOGLYPH DETECTED: {domain} -> resolves to IP")
        return registered

    def find_registered_homoglyphs(self, original_domain):
        """Generates variants and returns those that are currently registered."""
        variants = self.generate_variants(original_domain)
        # Remove original domain from being reported as a homoglyph of itself
        if original_domain in variants:
            variants.remove(original_domain)
            
        logging.info(f"Generated {len(variants)} visual variants for {original_domain}. Checking DNS concurrently with aiodns...")
        
        try:
            loop = asyncio.get_running_loop()
            # In a real environment like FastAPI, this would need to return the awaitable.
            # But for simplicity here, we assume it's running synchronously in a worker thread.
            return loop.run_until_complete(self._find_registered_async(variants))
        except RuntimeError:
            return asyncio.run(self._find_registered_async(variants))

if __name__ == "__main__":
    generator = HomoglyphGenerator()
    test_domain = "sbi.co.in"
    active_variants = generator.find_registered_homoglyphs(test_domain)
    print(f"Active Homoglyphs for {test_domain}: {active_variants}")
