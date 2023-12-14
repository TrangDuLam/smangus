import unicodedata

def atayal_to_pseudo_ipa(text):
    
    '''
    Generate pseudo IPA from Atayal text.
    Ref: https://en.wikipedia.org/wiki/Phonetic_symbols_in_Unicode
    '''
    
    ipa = ''
    
    for i in text :
        
        if i == "'" :
            ipa += unicodedata.normalize('NFKC', '\u0294')  #'ʔ'
        elif i == 'b' :
            ipa += unicodedata.normalize('NFKC', '\u03B2')  #'β'
        elif i == 'c' :
            ipa += unicodedata.normalize('NFKC', '\u02A6')  #'ʦ'
        elif i == 'g':
            ipa += unicodedata.normalize('NFKC', '\u0263')  #'ɣ'
        elif i == 'y' :
            ipa += unicodedata.normalize('NFKC', 'j')
        else : ipa += i
        
    return ipa