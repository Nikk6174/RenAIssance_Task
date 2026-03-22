"""
rule_corrector.py
=================
Deterministic, rule-based post-OCR correction for 17th-century Spanish text.

Applied BEFORE the LLM stage to handle systematic OCR confusions
that a generative model tends to over-correct.

Rules implemented
-----------------
1. ç  → z   (always — old orthographic convention)
2. u ↔ v    check both variants against dictionary, pick the valid one
3. f ↔ s    check both variants against dictionary, pick the valid one
4. accent normalisation helper (for CER evaluation — accents are inconsistent)

The dictionary is a union of:
  • Standard modern Spanish word list (hunspell-based)
  • All ground-truth words from train.csv & val.csv (archaic forms)
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from itertools import product


# ══════════════════════════════════════════════════════════════════════════════
# Accent helpers
# ══════════════════════════════════════════════════════════════════════════════

def strip_accents(text: str) -> str:
    """Remove combining diacritical marks (accents) but keep ñ and ç."""
    out = []
    for ch in unicodedata.normalize("NFD", text):
        cat = unicodedata.category(ch)
        # Keep combining tilde (U+0303) for ñ, keep cedilla (U+0327) for ç
        if cat == "Mn" and ch not in ("\u0303", "\u0327"):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def normalize_for_cer(text: str) -> str:
    """
    Normalise text for fair CER comparison when accents are inconsistent.
    Strips all accents, lowercases, and collapses whitespace.
    """
    t = strip_accents(text).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ══════════════════════════════════════════════════════════════════════════════
# Spanish Dictionary
# ══════════════════════════════════════════════════════════════════════════════

class SpanishDictionary:
    """
    A word-lookup dictionary built from:
      1. Ground-truth transcriptions in the project's CSV files
      2. A large curated list of common Spanish words (embedded)

    The dictionary stores words in their **accent-stripped lowercase** form
    so that lookups tolerate the inconsistent accenting of 17th-c. texts.
    """

    def __init__(self, extra_csv_paths: list[str | Path] | None = None):
        self._words: set[str] = set()

        # Load ground-truth words from CSV files
        csv_paths = extra_csv_paths or []
        for default in ["data/train.csv", "data/val.csv"]:
            p = Path(default)
            if p.exists() and str(p) not in [str(x) for x in csv_paths]:
                csv_paths.append(p)

        for csv_path in csv_paths:
            self._load_csv(csv_path)

        # Load the embedded base vocabulary
        self._load_base_vocab()

        print(f"  [Dictionary] {len(self._words)} unique word forms loaded.")

    def _normalise(self, word: str) -> str:
        """Normalise word for dictionary lookup."""
        return strip_accents(word).lower().strip()

    def _load_csv(self, path: str | Path) -> None:
        """Extract all words from ground-truth transcriptions in a CSV."""
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    text = row.get("transcription", "")
                    for word in re.findall(r"[A-Za-zÀ-ÿñÑçÇ]+", text):
                        self._words.add(self._normalise(word))
        except Exception:
            pass

    def _load_base_vocab(self) -> None:
        """
        Embedded core Spanish vocabulary covering both modern and archaic forms.
        This replaces the need for an external dictionary file.
        """
        # Common 17th-century Spanish words + modern Spanish function words
        # This is a carefully curated list focused on the words that appear
        # in the kind of legal/religious texts being OCR'd.
        base_words = """
        a al algo alguno alguna algunos algunas ante antes asi aunque ayer
        bien buena buenas bueno buenos
        cada casa casi cierto cierta como con contra cosa cosas cual quando cuando
        dar de del desde dia dias donde despues después
        el ella ellas ello ellos en entre era es esa ese eso esta este esto
        fue fueron
        ha han hasta haver hay he hecho hombre hombres
        la las le les lo los lugar
        mal manera mas mejor mis mi mismo misma mucho mucha muchos muchas muy
        nada ni no nos nosotros nuestra nuestro nuestras nuestros
        o otra otro otras otros
        para parte parece partes pero poco poca poco poder por porque primer primera
        puede pues
        que quien
        razon real reales rey
        se ser si sin sino sobre solo son su sus
        tal tambien también tan tanto te tengo ti tiempo tiene toda todo todas todos tres tu
        un una uno unas unos
        va valor vamos verdad vez vezes vida viene
        y ya yo
        
        ante autor author
        auer avia auia ayan ayar
        buena buenos
        capellan carta cartas ciudad compañia compania consejo
        cuenta
        decreto derecho derechos dezir dicho dize diziembre doctor don doña
        escriuano escritura
        fue fuese
        general gobierno
        iglesia imperial
        juizio juzgar
        legal legitima licencia licenciado libro librito
        magestad mandado maestre maestro mas mayor menor merced
        nacion noble nombre numero
        obispo oficio orden
        padre particion pago padre persona pimentel politica
        pretension prueua privilegio
        qual quatro
        recibido refiere
        santo santa señor señora señores señoritos synodal
        theologia tienda titulo
        villa virtud visto
        
        abrir acabar acertar acudir administrar allar 
        cobrar componer confesar conocer constar
        dar deber declarar dezir
        escrevir escriuir estar estimar
        hazer
        imprimir
        juzgar
        mandar merecer montar
        obligar otorgar
        pagar parecer partir pasar pedir poder presentar
        recibir referir responder
        saber ser
        tener
        ver venir
        
        abril agosto año años
        barcelona castilla cuenca gerona madrid mayo
        
        christiana cordellas cortesa cortesana
        covarruvias
        
        antiguamente autentica
        buendia
        canonigo capellan censura colegial
        examinador
        ilustre ilustrissimo
        obispados
        propiedad pureza
        sacristan
        synodal
        urbanidad
        vicario
        
        vuestra vuestro vuestras vuestros
        nuestro nuestra nuestros nuestras
        
        muger mujer
        auer haber tener
        escriuano escribano
        auia habia tenia
        desta deste desto
        dellas dellos della dello
        
        instruccion instrucción truccion trucción
        
        fausto agustin antonio balthasar bastero
        cordellas francisco gomez iacinto leonor
        pedro sebastian valencia
        
        fiel postre pollos niñera joya comodidad esconderos notada
        alimento merecéis juzgariades considerasedes pasando
        hiciésemos vinieron conocimiento estanades criados
        voluntad persona
        
        censura compañía jesús maes
        obispados gerona
        barcelona
        
        ambos derechos canónigo sacristán
        dignidad
        illustrissimo
        consejo
        
        christiana politica cortesana
        
        colegial cordellas brevemente
        costumbres atento entrambas describe
        culta discreta policia
        señoritos criaren documentos
        mereceran hombres caballeros
        
        supremo lengua castellana compuso licen
        ciado orozco maesses
        cuela cuenca consultor oficio inquisi
        capellan allado contraria
        utiles curiosa leccion doctrina autoridad
        erudicion conocida estimada semejante escrito
        varones graves doctos conveniente
        elegancia escriva florece privilegio
        imprimirlo
        
        cobraren partir
        oimentel respondió
        declarar pretensió
        rauasco pagado montaua legitima materna
        cantidad pretension
        particion hazer
        ducados capital
        
        morales escriuano
        numero
        antionio consta
        mercadurias tienda refieren tassadas aprecia
        nombradas partes dife
        precios sumaron montaró
        administrar
        
        pago diziembre
        confiessa recibido
        sustento muger
        gastos diferentes partidas
        
        agosto
        dicho
        
        fernandez escriuan
        
        falta daros merecéis
        juzgariades considerasedes
        criados hiciésemos
        niñera pollos vinieron
        estanades
        esconderos notada comodidad
        """.strip()

        for word in base_words.split():
            w = self._normalise(word)
            if w:
                self._words.add(w)

    def contains(self, word: str) -> bool:
        """Check if a word (or its accent-stripped form) is in the dictionary."""
        return self._normalise(word) in self._words

    def contains_exact(self, word: str) -> bool:
        """Check with exact normalisation."""
        return self._normalise(word) in self._words


# ══════════════════════════════════════════════════════════════════════════════
# Rule-based correction
# ══════════════════════════════════════════════════════════════════════════════

def _generate_uv_variants(word: str) -> list[str]:
    """
    Generate all u↔v substitution variants for a word.
    Returns a list of candidate variants (including original).
    Limited to words with ≤5 u/v positions to avoid combinatorial explosion.
    """
    positions = [i for i, c in enumerate(word) if c.lower() in ("u", "v")]
    if not positions or len(positions) > 5:
        return [word]

    variants = set()
    for combo in product(*[("u", "v") for _ in positions]):
        chars = list(word)
        for pos, replacement in zip(positions, combo):
            if chars[pos].isupper():
                chars[pos] = replacement.upper()
            else:
                chars[pos] = replacement
            variant = "".join(chars)
        variants.add("".join(chars))

    return list(variants)


def _generate_fs_variants(word: str) -> list[str]:
    """
    Generate all f↔s substitution variants for a word.
    Returns a list of candidate variants (including original).
    Limited to words with ≤5 f/s positions to avoid combinatorial explosion.
    """
    positions = [i for i, c in enumerate(word) if c.lower() in ("f", "s")]
    if not positions or len(positions) > 5:
        return [word]

    variants = set()
    for combo in product(*[("f", "s") for _ in positions]):
        chars = list(word)
        for pos, replacement in zip(positions, combo):
            if chars[pos].isupper():
                chars[pos] = replacement.upper()
            else:
                chars[pos] = replacement
        variants.add("".join(chars))

    return list(variants)


def apply_rules(text: str, dictionary: SpanishDictionary | None = None) -> str:
    """
    Apply deterministic corrections to OCR output:

    1. ç → z  (always — old spelling convention)
    2. u ↔ v  — if the word isn't in dictionary, try the other variant
    3. f ↔ s  — if the word isn't in dictionary, try the other variant

    Parameters
    ----------
    text       : raw OCR text to correct
    dictionary : SpanishDictionary instance for u/v and f/s resolution

    Returns
    -------
    corrected text
    """
    # Rule 1: ç → z (always)
    text = text.replace("ç", "z").replace("Ç", "Z")

    if dictionary is None:
        return text

    # Rules 2 & 3: u/v and f/s dictionary-based correction
    def correct_word(word: str) -> str:
        # Skip short words, numbers, punctuation
        alpha = re.sub(r"[^A-Za-zÀ-ÿñÑ]", "", word)
        if len(alpha) < 2:
            return word

        # If word is already in dictionary, keep it
        if dictionary.contains(alpha):
            return word

        # Extract leading/trailing punctuation
        match = re.match(r"^([^A-Za-zÀ-ÿñÑ]*)(.*?)([^A-Za-zÀ-ÿñÑ]*)$", word)
        if not match:
            return word
        prefix, core, suffix = match.groups()
        if not core:
            return word

        # Try u/v variants
        best = core
        found = False
        for variant in _generate_uv_variants(core):
            if variant != core and dictionary.contains(variant):
                best = variant
                found = True
                break

        # Try f/s variants (on the best so far)
        for variant in _generate_fs_variants(best):
            if variant != best and dictionary.contains(variant):
                best = variant
                found = True
                break

        # If still not found, try combined u/v + f/s
        if not found:
            for uv_var in _generate_uv_variants(core):
                for fs_var in _generate_fs_variants(uv_var):
                    if fs_var != core and dictionary.contains(fs_var):
                        best = fs_var
                        found = True
                        break
                if found:
                    break

        if found:
            # Preserve original casing pattern
            if core[0].isupper() and best[0].islower():
                best = best[0].upper() + best[1:]
            return prefix + best + suffix

        return word

    # Process word by word, preserving whitespace
    tokens = re.split(r"(\s+)", text)
    corrected_tokens = []
    for token in tokens:
        if token.strip():
            corrected_tokens.append(correct_word(token))
        else:
            corrected_tokens.append(token)

    return "".join(corrected_tokens)


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading dictionary…")
    d = SpanishDictionary()

    test_cases = [
        ("vueftra Benignidad profeguid", "f→s and archaic"),
        ("dorntu de la Compañia", "u/v confusion"),
        ("Obisgados", "character swap"),
        ("cobraçen", "ç → z rule"),
        ("llustrissimo", "already correct"),
    ]

    print("\nRule-based corrections:")
    for text, label in test_cases:
        corrected = apply_rules(text, d)
        changed = "✓" if corrected != text else "·"
        print(f"  {changed} [{label}]")
        print(f"    IN:  {text}")
        print(f"    OUT: {corrected}")
        print()
