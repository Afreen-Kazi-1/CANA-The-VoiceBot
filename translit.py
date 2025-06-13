# import the module
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# the text to be transliterated
text = "Apa sabhii kaa yahaan svaagat hai."

# printing the transliterated text
print(transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI))