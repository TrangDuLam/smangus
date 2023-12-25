import unicodedata

def atayal_to_pseudo_ipa(text):
    
    '''
    Generate pseudo IPA from Atayal text.
    Ref: https://en.wikipedia.org/wiki/Phonetic_symbols_in_Unicode
    '''
    
    preIPA = ''
    
    for i in text :
        
        if i == "'" :
            preIPA += unicodedata.normalize('NFKC', '\u0294')  #'ʔ'
        elif i == 'b' :
            preIPA += unicodedata.normalize('NFKC', '\u03B2')  #'β'
        elif i == 'c' :
            preIPA += unicodedata.normalize('NFKC', '\u02A6')  #'ʦ'
        elif i == 'g':
            preIPA += unicodedata.normalize('NFKC', '\u0263')  #'ɣ'
        elif i == 'y' :
            preIPA += unicodedata.normalize('NFKC', 'j')
        else : preIPA += i
        
    needToConvert_ng = 'n' + unicodedata.normalize('NFKC', '\u0263')  #'ŋ'
    
    for i in range(len(preIPA)) :
        preIPA = preIPA.replace(needToConvert_ng, unicodedata.normalize('NFKC', '\u014B'))
        
    con_phm = []
        
    for j in range(len(preIPA)) :
            
            if preIPA[j] not in 'aeiouj' and preIPA[j] != ' ': 
                # Consonants
                if lastChar == 'C':
                    con_phm.append(unicodedata.normalize('NFKC', '\u0259')) # push mid vowel
                    con_phm.append(preIPA[j]) # push current consonant
                else :
                    con_phm.append(preIPA[j])
            
                lastChar = 'C'
                
            else : 
                # Vowels
                con_phm.append(preIPA[j])
                lastChar = 'V'

    return ''.join(con_phm)