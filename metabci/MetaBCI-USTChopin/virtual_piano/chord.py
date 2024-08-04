import copy

chord_list = {
    # 单音
    '': ' ',
    # 单音程
    'b': ' with minor 2nd',
    'c': ' with major 2nd',
    'd': ' with minor 3rd',
    'e': ' with major 3rd',
    'f': ' with perfect 4th',
    'g': ' with augmented 4th',
    'i': ' with minor 6th',
    'j': ' with major 6th',
    'k': ' with minor 7th',
    'l': ' with major 7th',
    # 三和弦
    'eh': '',
    'dh': 'm',
    'ei': 'aug',
    'dg': 'dim',
    'h': '5',
    # 七和弦
    'ehl': 'maj7',
    'el': 'maj7(omit5)',
    'hl': 'maj7(omit3)',
    'dhk': 'm7',
    'dk': 'm7(omit5)',
    'hk': '(m)7(omit3)',
    'ek': '7(omit5)',
    'ehk': '7',
    'eil': 'maj7#5',
    'il': 'maj7#5(omit3)',
    'dhl': 'mM7',
    'dl': 'mM7(omit5)',
    'dgj': 'dim7',
    'dj': 'm6(omit5)',
    'gj': 'dim7(omit3)',
    'dgk': 'm7b5',
    # 含有九音的和弦
    'cehl': 'maj9',
    'ceh': 'add9',
    'cel': 'maj9(omit5)',
    'chl': 'maj9(omit3)',
    'cl': 'maj9(omit3,5)',
    'ch': 'sus2',
    'ce': 'add9(omit5)',
    'cdhk': 'm9',
    'cdh': 'madd9',
    'cdk': 'm9(omit5)',
    'chk': '(m)9(omit3)',
    'ck': '(m)9(omit3,5)',
    'cd': 'madd9(omit5)',
    'cehk': '9',
    'cek': '9(omit5)',
    'behk': '7b9',
    'bek': '7b9(omit5)',
    'bhk': '7b9(omit3)',
    'bk': '7b9(omit3,5)',
    'beh': 'addb9',
    'be': 'addb9(omit5)',
    'bh': '5addb9',
    'bdh': 'maddb9',
    'bd:': 'maddb9(omit5)',
    'deh': 'add#9',
    'dehk': '7#9',
    'dek': '7#9(omit5)',
    'cdgk': 'm9b5',
    'bdgk': 'm7(b5,b9)',
    'bgk': '(m)7(b5,b9)(omit3)',
    'cdgj': 'dim7(9)',
    'cgj': 'dim7(9)(omit3)',
    # 三和弦 add11
    'efh': 'add11',
    'dfh': 'madd11',
    'efi': 'aug(11)',
    'dfg': 'dim(11)',
    'fh': 'sus4',
    # 七和弦 add11
    'efhl': 'maj7(11)',
    'efl': 'maj7(11)(omit5)',
    'fhl': 'maj7sus4',
    'fl': 'maj7sus4(omit5)',
    'dfhk': 'm7(11)',
    'dfk': 'm7(11)(omit5)',
    'fhk': '7sus4',
    'fk': '7sus4(omit5)',
    'efk': '7(11)(omit5)',
    'efhk': '7(11)',
    'efil': 'maj7(#5,11)',
    'fil': 'maj7(#5,11)(omit3)',
    'dfhl': 'mM7(11)',
    'dfl': 'mM7(11)(omit5)',
    'dfgj': 'dim7(11)',
    'dfj': 'dim7(11)(omit5)',
    'fgj': 'dim7(11)(omit3)',
    'dfgk': 'm7(b5,11)',
    'fgk': 'm7(b5,11)(omit3)',
    # 含有九音、十一音的和弦
    'cefhl': 'maj11',
    'cefh': '(9,11)',
    'cefl': 'maj11(omit5)',
    'cfhl': 'maj9sus4',
    'cfl': 'maj11(omit3,5)',
    'cfh': 'sus4add9',
    'cef': '(9,11)(omit5)',
    'cdfhk': 'm11',
    'cdfh': 'm(9,11)',
    'cdfk': 'm11(omit5)',
    'cfhk': '(m)11(omit3)',
    'cfk': '(m)11(omit3,5)',
    'cdf': 'm(9,11)(omit5)',
    'cefhk': '11',
    'cefk': '11(omit5)',
    'befhk': '11b9',
    'befk': '11b9(omit5)',
    'bfhk': '11b9(omit3)',
    'bfk': '11b9(omit3,5)',
    'befh': '(b9,11)',
    'bdfh': 'm(b9,11)',
    'defhk': '11#9',
    'defk': '11#9(omit5)',
    'bef': '(b9,11)(omit5)',
    'bfh': '5(b9,11)',
    'bdf:': 'm(b9,11)(omit5)',
    'cdfgk': 'm11b5',
    'cfgk': 'm11b5(omit3)',
    'bdfgk': 'm11(b5,b9)',
    'bfgk': 'm11(b5,b9)(omit3)',
    'cdfgj': 'dim7(9,11)',
    'cfgj': 'dim7(9,11)(omit3)',
    # 三和弦 add#11
    'egh': 'add#11',
    'dgh': 'madd#11',
    'egi': 'aug(#11)',
    'gh': '5(#11)',
    # 七和弦 add#11
    'eghl': 'maj7(#11)',
    'egl': 'maj7(#11)(omit5)',
    'ghl': 'maj7(#11)(omit3)',
    'gl': 'maj7(#11)(omit3,5)',
    'dghk': 'm7(#11)',
    'ghk': '7#11(omit3)',
    'gk': '(m)7b5(omit3)',
    'egk': '7(#11)(omit5)',
    'eghk': '7(#11)',
    'egil': 'maj7(#5,#11)',
    'gil': 'maj7(#5,#11)(omit3)',
    'dghl': 'mM7(#11)',
    'dgl': 'mM7(#11)(omit5)',
    # 含有九音、#十一音的和弦
    'ceghl': 'maj9#11',
    'cegh': '(9,#11)',
    'cegl': 'maj9#11(omit5)',
    'cghl': 'maj9#11(omit3)',
    'cgl': 'maj7#11(omit3,5)',
    'cgh': '5(9,#11)',
    'ceg': '(9,#11)(omit5)',
    'cdghk': 'm9(#11)',
    'cdgh': 'm(9,#11)',
    'cghk': '9#11(omit3)',
    'cgk': '(m)9b5(omit3)',
    'cdg': 'm(9,#11)(omit5)',
    'ceghk': '9#11',
    'cegk': '9#11(omit5)',
    'beghk': '7(b9,#11)',
    'begk': '7(b9,#11)(omit5)',
    'bghk': '7(b9,#11)(omit3)',
    'begh': '(b9,#11)',
    'bdgh': 'm(b9,#11)',
    'deghk': '7(#9,#11)',
    'degk': '7(#9,#11)(omit5)',
    # 三和弦 add13(b13)
    'ehj': 'add6',
    'ej': 'add6(omit5)',
    'dhj': 'madd6',
    'eij': 'augadd6',
    'hj': '5add6',
    'ehi': 'addb6',
    'dhi': 'maddb6',
    'hi': '5addb6',
    # 七和弦 add13(b13)
    'ehjl': 'maj7(13)',
    'ejl': 'maj7(13)(omit5)',
    'hjl': 'maj7(13)(omit3)',
    'dhjk': 'm7(13)',
    'djk': 'm7(13)(omit5)',
    'hjk': '(m)7(13)(omit3)',
    'ejk': '7(13)(omit5)',
    'ehjk': '7(13)',
    'eijl': 'maj7(#5,13)',
    'ijl': 'maj7(#5,13)(omit3)',
    'dhjl': 'mM7(13)',
    'djl': 'mM7(13)(omit5)',
    'dgjk': 'm7(b5,13)',
    'gjk': '(m)7(b5,13)(omit3)',
    'ehil': 'maj7(b13)',
    'hil': 'maj7(b13)(omit3)',
    'dhik': 'm7(b13)',
    'dik': 'm7(b13)(omit5)',
    'hik': '(m)7(b13)(omit3)',
    'eik': '7(b13)(omit5)',
    'ehik': '7(b13)',
    'dhil': 'mM7(b13)',
    'dil': 'mM7(b13)(omit5)',
    'dgik': 'm7(b5,b13)',
    'gik': '(m)7(b5,b13)(omit3)',
    'dgij': 'dim7(b13)',
    'dij': 'dim7(b13)(omit5)',
    'gij': 'dim7(b13)(omit3)',
    # 含有九音的和弦 add13(b13)
    'cehjl': 'maj9(13)',
    'cehj': '6/9',
    'cejl': 'maj9(13)(omit5)',
    'chjl': 'maj9(13)(omit3)',
    'cjl': 'maj9(13)(omit3,5)',
    'chj': '6/9(omit3)',
    'cej': '6/9(omit5)',
    'cdhjk': 'm9(13)',
    'cdhj': 'm(9,13)',
    'cdjk': 'm9(13)(omit5)',
    'chjk': '(m)9(13)(omit3)',
    'cjk': '(m)9(13)(omit3,5)',
    'cdj': 'm(9,13)(omit5)',
    'cehjk': '9(13)',
    'cejk': '9(13)(omit5)',
    'behjk': '7(b9,13)',
    'bejk': '7(b9,13)(omit5)',
    'bhjk': '7(b9,13)(omit3)',
    'bjk': '7(b9,13)(omit3,5)',
    'behj': '(b9,13)',
    'bej': '(b9,13)(omit5)',
    'bhj': '5(b9,13)',
    'bdhj': 'm(b9,13)',
    'bdj:': 'm(b9,13)(omit5)',
    'dehjk': '7(#9,13)',
    'dejk': '7(#9,13)(omit5)',
    'cehil': 'maj9(b13)',
    'cehi': '(9,b13)',
    'ceil': 'maj9(b13)(omit5)',
    'chil': 'maj9(b13)(omit3)',
    'cil': 'maj9(b13)(omit3,5)',
    'chi': '(m)(9,b13)(omit3)',
    'cdhik': 'm9(b13)',
    'cdhi': 'm(9,b13)',
    'chik': '(m)9(b13)(omit3)',
    'cik': '(m)9(b13)(omit3,5)',
    'cehik': '9(b13)',
    'ceik': '9(b13)(omit5)',
    'behik': '7(b9,b13)',
    'beik': '7(b9,b13)(omit5)',
    'bhik': '(m)7(b9,b13)(omit3)',
    'bik': '(m)7(b9,b13)(omit3,5)',
    'bdik': 'm7(b9,b13)(omit5)',
    'bdhik': 'm7(b9,b13)',
    'behi': '(b9,b13)',
    'bhi': '5(b9,b13)',
    'bdhi': 'm(b9,b13)',
    'dehik': '7(#9,b13)',
    'deik': '7(#9,b13)(omit5)',
    'cdgij': 'dim7(9,b13)',
    'cdij': 'dim7(9,b13)(omit5)',
    'cgij': 'dim7(9,b13)(omit3)',
    # 三和弦 add11,13,b13
    'efhj': '(11,13)',
    'dfhj': 'm(11,13)',
    'fhj': '(m)(11,13)(omit3)',
    'efhi': '(11,b13)',
    'dfhi': 'm(11,b13)',
    'fhi': '(m)(11,b13)(omit3)',
    # 七和弦 add11,13,b13
    'efhjl': 'maj7(11,13)',
    'efjl': 'maj7(11,13)(omit5)',
    'fhjl': 'maj7(11,13)(omit3)',
    'fjl': 'maj7(11,13)(omit3,5)',
    'dfhjk': 'm7(11,13)',
    'dfjk': 'm7(11,13)(omit5)',
    'fhjk': '(m)7(11,13)(omit3)',
    'fjk': '(m)7(11,13)(omit3,5)',
    'efjk': '7(11,13)(omit5)',
    'efhjk': '7(11,13)',
    'efijl': 'maj7(#5,11,13)',
    'fijl': 'maj7(#5,11,13)(omit3)',
    'dfhjl': 'mM7(11,13)',
    'dfjl': 'mM7(11,13)(omit5)',
    'dfgjk': 'm7(b5,11,13)',
    'fgjk': 'm7(b5,11,13)(omit3)',
    'efhil': 'maj7(11,b13)',
    'fhil': 'maj7(11,b13)(omit3)',
    'dfhik': 'm7(11,b13)',
    'dfik': 'm7(11,b13)(omit5)',
    'fhik': '(m)7(11,b13)(omit3)',
    'fik': '(m)7(11,b13)(omit3,5)',
    'efik': '7(11,b13)(omit5)',
    'efhik': '7(11,b13)',
    'dfhil': 'mM7(11,b13)',
    'dfil': 'mM7(11,b13)(omit5)',
    'dfgik': 'm7(b5,11,b13)',
    'fgik': 'm7(b5,11,b13)(omit3)',
    'dfgij': 'dim7(11,b13)',
    'dfij': 'dim7(11,b13)(omit5)',
    'fgij': 'dim7(11,b13)(omit3)',
    # 含有九音、十一音、#b十三音的和弦
    'cefhjl': 'maj13',
    'cefhj': '(9,11,13)',
    'cefjl': 'maj13(omit5)',
    'cfhjl': 'maj13(omit3)',
    'cfjl': 'maj13(omit3,5)',
    'cfhj': '5(9,11,13)',
    'cefj': '(9,11,13)(omit5)',
    'cdfhjk': 'm13',
    'cdfhj': 'm(9,11,13)',
    'cdfjk': 'm13(omit5)',
    'cfhjk': '(m)13(omit3)',
    'cfjk': '(m)13(omit3,5)',
    'cdfj': 'm(9,11,13)(omit5)',
    'cefhjk': '13',
    'cefjk': '13(omit5)',
    'befhjk': '13b9',
    'befjk': '13b9(omit5)',
    'bfhjk': '13b9(omit3)',
    'bfjk': '13b9(omit3,5)',
    'befhj': '(b9,11,13)',
    'bdfhj': 'm(b9,11,13)',
    'defhjk': '13#9',
    'defjk': '13#9(omit5)',
    'befj': '(b9,11,13)(omit5)',
    'bfhj': '5(b9,11,13)',
    'bdfj:': 'm(b9,11,13)(omit5)',
    'cefhil': 'maj11(b13)',
    'cefhi': '(9,11,b13)',
    'cefil': 'maj11(b13)(omit5)',
    'cfhil': 'maj11(b13)(omit3)',
    'cfil': 'maj11(b13)(omit3,5)',
    'cfhi': '5(9,11,b13)',
    'cefi': '(9,11,b13)(omit5)',
    'cdfhik': 'm11(b13)',
    'cdfhi': 'm(9,11,b13)',
    'cdfik': 'm11(b13)(omit5)',
    'cfhik': '(m)11(b13)(omit3)',
    'cfik': '(m)11(b13)(omit3,5)',
    'cdfi': 'm(9,11,b13)(omit5)',
    'cefhik': '11(b13)',
    'cefik': '11(b13)(omit5)',
    'befhik': '11(b9,b13)',
    'befik': '11(b9,b13)(omit5)',
    'bfhik': '11(b9,b13)(omit3)',
    'bfik': '11(b9,b13)(omit3,5)',
    'befhi': '(b9,11,b13)',
    'bdfhi': 'm(b9,11,b13)',
    'defhik': '11(#9,b13)',
    'defik': '11(#9,b13)(omit5)',
    'befi': '(b9,11,b13)(omit5)',
    'bfhi': '5(b9,11,b13)',
    'bdfi:': 'm(b9,11,b13)(omit5)',
    'cdfgij': 'dim7(9,11,b13)',
    'cdfij': 'dim7(9,11,b13)(omit5)',
    'cfgij': 'dim7(9,11,b13)(omit3)',
    # 三和弦 add#11
    'eghj': '(#11,13)',
    'dghj': 'm(#11,13)',
    'egij': 'aug(#11,13)',
    'ghj': '5(#11,13)',
    'eghi': '(#11,b13)',
    'dghi': 'm(#11,b13)',
    'ghi': '5(#11,b13)',
    # 七和弦 add#11
    'eghjl': 'maj7(#11,13)',
    'egjl': 'maj7(#11,13)(omit5)',
    'ghjl': 'maj7(#11,13)(omit3)',
    'gjl': 'maj7(#11,13)(omit3,5)',
    'dghjk': 'm7(#11,13)',
    'ghjk': '7(#11,13)(omit3)',
    'egjk': '7(#11,13)(omit5)',
    'eghjk': '7(#11,13)',
    'egijl': 'maj7(#5,#11,13)',
    'gijl': 'maj7(#5,#11,13)(omit3)',
    'dghjl': 'mM7(#11,13)',
    'dgjl': 'mM7(#11,13)(omit5)',
    'eghil': 'maj7(#11,b13)',
    'ghil': 'maj7(#11,b13)(omit3)',
    'dghik': 'm7(#11,b13)',
    'ghik': '7(#11,b13)(omit3)',
    'egik': '7(#11,b13)(omit5)',
    'eghik': '7(#11,b13)',
    'dghil': 'mM7(#11,b13)',
    'dgil': 'mM7(#11,b13)(omit5)',
    # 含有九音、#十一音的和弦
    'ceghjl': 'maj13#11',
    'ceghj': '(9,#11,13)',
    'cegjl': 'maj13#11(omit5)',
    'cghjl': 'maj13#11(omit3)',
    'cgjl': 'maj13#11(omit3,5)',
    'cghj': '5(9,#11,13)',
    'cegj': '(9,#11,13)(omit5)',
    'cdghjk': 'm13(#11)',
    'cdghj': 'm(9,#11,13)',
    'cdgjk': 'm13(#11)(omit5)',
    'cghjk': '13(#11)(omit3)',
    'cgjk': '13(#11)(omit3,5)',
    'ceghjk': '13(#11)',
    'cegjk': '13(#11)(omit5)',
    'beghjk': '13(b9,#11)',
    'begjk': '13(b9,#11)(omit5)',
    'bghjk': '13(b9,#11)(omit3)',
    'bgjk': '13(b9,#11)(omit3,5)',
    'beghj': '(b9,#11,13)',
    'bdghj': 'm(b9,#11,13)',
    'deghjk': '13(#9,#11)',
    'degjk': '13(#9,#11)(omit5)',
    'ceghil': 'maj9(#11,b13)',
    'ceghi': '(9,#11,b13)',
    'cegil': 'maj9(#11,b13)(omit5)',
    'cghil': 'maj9(#11,b13)(omit3)',
    'cgil': 'maj7(#11,b13)(omit3,5)',
    'cghi': '5(9,#11,b13)',
    'cegi': '(9,#11,b13)(omit5)',
    'cdghik': 'm9(#11,b13)',
    'cdghi': 'm(9,#11,b13)',
    'cdgik': 'm9(b5,b13)',
    'cghik': '9(#11,b13)(omit3)',
    'cgik': '9(#11,b13)(omit3,5)',
    'cdgi': 'm(9,#11,b13)(omit5)',
    'ceghik': '9(#11,b13)',
    'cegik': '9(#11,b13)(omit5)',
    'beghik': '7(b9,#11,b13)',
    'begik': '7(b9,#11,b13)(omit5)',
    'bghik': '7(b9,#11,b13)(omit3)',
    'bgik': '7(b9,#11,b13)(omit3,5)',
    'beghi': '(b9,#11,b13)',
    'bdghi': 'm(b9,#11,b13)',
    'deghik': '7(#9,#11,b13)',
    'degik': '7(#9,#11,b13)(omit5)',
    'bdgik': 'm7(b9,#11,b13)(omit5)',
    'bdghik': 'm7(b9,#11,b13)',
    # 变化属和弦补充（同时含有自然延伸音与变化延伸音等）
    'bdehk': '7(b9,#9)',
    'bdek': '7(b9,#9)(omit5)',
    'bdeh': '(b9,#9)',
    'bde': '(b9,#9)(omit5)',
    'bdefhk': '11(b9,#9)',
    'bdefk': '11(b9,#9)(omit5)',
    'bdefh': '(b9,#9,11)',
    'bdef': '(b9,#9,11)(omit5)',
    'bdeghk': '7(b9,#9,#11)',
    'bdegk': '7(b9,#9,#11)(omit5)',
    'bdegh': '(b9,#9,#11)',
    'bdeg': '(b9,#9,#11)(omit5)',
    # 13
    'bdefhjk': '13(b9,#9)',
    'bdefjk': '13(b9,#9)(omit5)',
    'bdefhj': '(b9,#9,11,13)',
    'bdefj': '(b9,#9,11,13)(omit5)',
    'bdeghjk': '13(b9,#9,#11)',
    'bdegjk': '13(b9,#9,#11)(omit5)',
    'bdeghj': '(b9,#9,#11,13)',
    'bdegj': '(b9,#9,#11,13)(omit5)',
    # b13
    'bdefhik': '11(b9,#9,b13)',
    'bdefik': '11(b9,#9,b13)(omit5)',
    'bdefhi': '(b9,#9,11,b13)',
    'bdefi': '(b9,#9,11,b13)(omit5)',
    'bdeghik': '7(b9,#9,#11,b13)',
    'bdegik': '7(b9,#9,#11,b13)(omit5)',
    'bdeghi': '(b9,#9,#11,b13)',
    'bdegi': '(b9,#9,#11,b13)(omit5)'
}

# 在特定的和弦排列中预设根音，仅包含内声部和高音与低音的音程关系，值为（从低音往上数）的根音
chord_presets = {
    'eil': 0,
    'cfh': 0,
    'dif': 2
}

# 和弦属性不同时，音名略有差异（有习惯的因素）
pitch_name = ['C', 'C#(Db)', 'D', 'D#(Eb)', 'E', 'F', 'F#(Gb)', 'G', 'G#(Ab)', 'A', 'A#(Bb)', 'B']
pitch_name_if_single = ['C', 'C#(Db)', 'D', 'D#(Eb)', 'E', 'F', 'F#(Gb)', 'G', 'G#(Ab)', 'A', 'A#(Bb)', 'B']
pitch_name_if_augmented = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
pitch_name_if_major = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
pitch_name_if_diminished = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
pitch_name_if_minor = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

priority_each_note = [0, -1, -1, 1, 1, -1, -1, 1, -5, -1, 0, 0]  # 各音程权重

# 把所有的音程关系转化为字母，最后的和弦为字符串，方便查表
tocode = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']


# 输入音名返回在一个八度内的排序（仅含CDEFGAB，不含升降号）
def get_order(c):
    if c == 'A':
        return 5
    elif c == 'B':
        return 6
    else:
        return ord(c) - 67


# 返回和弦标记字符串（不包含根音）
def to_chord(input_intervals):
    global tocode
    intervals_str = ''
    arranged_intervals = sorted(input_intervals)
    for i in arranged_intervals:
        if (tocode[i] not in intervals_str) and i != 0:
            intervals_str += tocode[i]
    if intervals_str not in chord_list:
        return '...'
    return chord_list[intervals_str]


# 按优先级确定和弦根音（不一定准确，需要自行添加规则）
def get_chord(key_note, on_sustain, major_key, root_decide):
    global pitch_name_if_single
    global pitch_name_if_augmented
    global pitch_name_if_major
    global pitch_name_if_diminished
    global pitch_name_if_minor
    global pitch_name
    global priority_each_note
    note_on_sheet = []

    # 根据调性来规范音名
    if major_key == 'Unsettled':
        pitch_name = ['C', 'C#(Db)', 'D', 'D#(Eb)', 'E', 'F', 'F#(Gb)', 'G', 'G#(Ab)', 'A', 'A#(Bb)', 'B']
        pitch_name_if_single = ['C', 'C#(Db)', 'D', 'D#(Eb)', 'E', 'F', 'F#(Gb)', 'G', 'G#(Ab)', 'A', 'A#(Bb)', 'B']
        pitch_name_if_augmented = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_major = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_diminished = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_name_if_minor = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if major_key == 'C':
        pitch_name_if_single = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#(Ab)', 'A', 'Bb', 'B']
        pitch_name_if_augmented = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_major = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_diminished = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
        pitch_name_if_minor = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    if major_key == 'Db':
        pitch_name_if_single = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_augmented = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_major = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_diminished = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_minor = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    if major_key == 'D':
        pitch_name_if_single = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#(Bb)', 'B']
        pitch_name_if_augmented = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
        pitch_name_if_major = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
        pitch_name_if_diminished = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_name_if_minor = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if major_key == 'Eb':
        pitch_name_if_single = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_augmented = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_major = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_diminished = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_minor = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    if major_key == 'E':
        pitch_name_if_single = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_name_if_augmented = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_name_if_major = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_name_if_diminished = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_name_if_minor = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if major_key == 'F':
        pitch_name_if_single = ['C', 'C#(Db)', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_augmented = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_major = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_diminished = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_minor = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    if major_key == 'Gb':
        pitch_name_if_single = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']
        pitch_name_if_augmented = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']
        pitch_name_if_major = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']
        pitch_name_if_diminished = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']
        pitch_name_if_minor = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']
    if major_key == 'G':
        pitch_name_if_single = ['C', 'C#', 'D', 'D#(Eb)', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_augmented = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_major = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_diminished = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_minor = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    if major_key == 'Ab':
        pitch_name_if_single = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_augmented = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_major = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_diminished = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_minor = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    if major_key == 'A':
        pitch_name_if_single = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
        pitch_name_if_augmented = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
        pitch_name_if_major = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
        pitch_name_if_diminished = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
        pitch_name_if_minor = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    if major_key == 'Bb':
        pitch_name_if_single = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#(Gb)', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_augmented = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_major = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_diminished = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        pitch_name_if_minor = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    if major_key == 'B':
        pitch_name_if_single = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_name_if_augmented = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_name_if_major = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_name_if_diminished = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pitch_name_if_minor = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    pitch_name_temp = copy.deepcopy(pitch_name_if_single)

    # 将所有音（包括踏板）加入inp_unsorted中
    inp_unsorted = {}
    for i in key_note:
        inp_unsorted[i + 9] = 1
    for i in on_sustain:
        inp_unsorted[i - 12] = 0

    # root为输入的指定和弦根音编号（以A开始），root_virtual为以C开始在一个八度内的编号
    if root_decide == -1:
        root_virtual = -1
    else:
        root_virtual = (root_decide - 3) % 12  # 预先决定根音，而非后期判断权重
        if root_virtual not in inp_unsorted:
            inp_unsorted[root_virtual + 108] = -1  # 把根音加进来，以判断和弦，+96是为了防止其取代低音位置

    # 将所有音排序，元素为note编号（以C开始）
    inp_dict = dict(sorted(inp_unsorted.items(), key=lambda x: x[0]))
    inp = list(inp_dict.keys())

    # 如果没有预先决定根音，那么看看预设里面有没有该和弦排列
    if root_virtual == -1 and len(inp) > 0:
        judge_str = ''  # 判断预设字符串
        top = inp[len(inp) - 1] % 12  # 最高音
        bas = inp[0] % 12  # 最低音
        inner_note_intervals = []  # 内声部所有音与低音音程关系
        for i in inp:
            inner_note = i % 12  # 单个音
            if inner_note != top and inner_note != bas:
                if ((inner_note - bas) % 12) not in inner_note_intervals:
                    inner_note_intervals.append((inner_note - bas) % 12)
        inner_note_intervals.sort()  # 内声部所有音（在一个8度范围内）做一个排序
        # inner_note_intervals is already in order and % 12
        for i in inner_note_intervals:
            judge_str += tocode[i]
        judge_str += tocode[top - bas]  # 仅包含内声部、最高音与低音的音程关系
        # 加入最低音和最高音，此时inner_note_intervals变量为【低音+内声部（有序）+高音】音程
        inner_note_intervals.insert(0, 0)
        inner_note_intervals.append(top - bas)
        if judge_str in chord_presets:
            root_virtual = (inner_note_intervals[chord_presets[judge_str]] + bas) % 12  # 预先决定的根音在preset里能找到

    # 对所有和弦结果设定优先级
    all_result = []  # 所有可能出现的和弦判断结果，优先级不同
    bass = 0
    if len(inp) > 0:
        bass = inp[0]  # 低音

    # 尝试将每个音作为根音，将其他音与根音音程关系存入intervals
    for root in inp:
        priority = 10  # 初始优先级为10
        intervals = []  # 根音与其它音的音程关系
        for others in inp:
            interval = (others - root) % 12  # 单个音程
            if interval not in intervals:
                intervals.append(interval)
                priority += priority_each_note[interval]  # 各音"权重"决定优先级

        # 获得和弦名称（不含根音名称）
        chord_name = to_chord(intervals)

        # 看待转位和弦时，一些特殊的习惯（仅为习惯，不保证严谨）
        bass_root_length = (bass - root) % 12  # 低音与根音音程距离

        # 1：九，十一，十三音不常作为低音
        if (bass_root_length != 0 and bass_root_length != 3 and bass_root_length != 4
                and bass_root_length != 7 and bass_root_length != 10 and bass_root_length != 11):
            priority -= 5

        # 2：如果小和弦有降九音，那么它的第一转位按照另一种和弦来看待
        if 1 in intervals and bass_root_length == 3:
            priority -= 5

        # 3：如果有四音没三音，那么它的第二转位按照另一种和弦看待
        if 4 not in intervals and 3 not in intervals and 5 in intervals and bass_root_length == 7:
            priority -= 5

        # 4：如果小和弦有七音，那么它的第一转位按照另一种和弦看待
        if 10 in intervals and bass_root_length == 3:
            priority -= 5

        # 5：如果小和弦没有五音，却有十一音和十三音，最好按照另一种和弦看待
        if 3 in intervals and 7 not in intervals and 5 in intervals and 9 in intervals:
            priority -= 3

        # 6：十一，十三音在里面第二转位较少
        if 5 in intervals and 9 in intervals and bass_root_length == 7:
            priority -= 3

        # 7：九，十三音在里面第二转位较少
        if 2 in intervals and 9 in intervals and bass_root_length == 7:
            priority -= 3

        # 8：九音，十一音在里面第三转位较少
        if 2 in intervals and 5 in intervals and bass_root_length == 10:
            priority -= 3

        # 9：有十一音，很少进行第三转位
        if 5 in intervals and bass_root_length == 10:
            priority -= 2

        # 10：没名称的降低优先级
        if '...' in chord_name:
            priority -= 7

        om = chord_name.find('(omit')
        if om != -1:
            chord_name_nomit = chord_name[:om]
        else:
            chord_name_nomit = chord_name
        # 这里采用判断字符串来判断和弦属性（由于写chord_list没另设属性值，只能这样了哈哈哈）
        if chord_name_nomit == ' ':
            root_name = pitch_name_if_single[root % 12]
        elif 'aug' in chord_name_nomit:
            # Augmented
            root_name = pitch_name_if_augmented[root % 12]
        elif ('maj' in chord_name_nomit) or ('m' not in chord_name_nomit):
            # Major or Dominant
            root_name = pitch_name_if_major[root % 12]
        elif 'dim' in chord_name_nomit:
            # Diminished
            root_name = pitch_name_if_diminished[root % 12]
        else:
            # Minor
            root_name = pitch_name_if_minor[root % 12]
        res = root_name + chord_name
        all_result.append([res, priority, root, root_name, chord_name_nomit])

    # 从所有结果里找到优先级最高的
    decided_res = ''  # 包含omit标识
    decided_root = 0
    decided_root_name = ''
    decided_chord_name = ''  # 没有omit标识
    if root_virtual == -1:
        # auto decide root
        i = -9999
        for j in all_result:
            if i < j[1]:
                i = j[1]
                decided_res = j[0]
                decided_root = j[2]
                decided_root_name = j[3]
                decided_chord_name = j[4]
    else:
        # decided root
        for j in all_result:
            if root_virtual == j[2] % 12:
                decided_res = j[0]
                decided_root = j[2]
                decided_root_name = j[3]
                decided_chord_name = j[4]

    if len(decided_root_name) >= 3:
        # 不确定记谱
        pitch_name_temp = ['C', 'C#(Db)', 'D', 'D#(Eb)', 'E', 'F', 'F#(Gb)', 'G', 'G#(Ab)', 'A', 'A#(Bb)', 'B']
    if decided_chord_name == '...':
        pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    else:
        if decided_root_name == 'C':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['C', 'Db', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'Db', 'D', 'D#', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'Bbb', 'Bb', 'B']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'E', 'F',
                                   'F#' if '#11' in decided_chord_name else 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        if decided_root_name == 'Db':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['C', 'Db', 'Ebb', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'Db', 'Ebb', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'Bbb', 'Bb', 'Cb']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'Db', 'Ebb', 'Eb', 'Fb', 'F', 'Gb', 'Abb', 'Ab', 'Bbb', 'Cbb', 'Cb']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'Db', 'Ebb', 'Eb', 'Fb', 'F', 'Gb',
                                   'G' if '#11' in decided_chord_name else 'Abb', 'Ab', 'Bbb', 'Bb', 'Cb']
        if decided_root_name == 'C#':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['B#', 'C#', 'D', 'D#', 'Dx', 'E#', 'F#', 'Fx', 'G#', 'Gx', 'A#', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['B#', 'C#', 'D', 'D#', 'Dx', 'E#', 'F#', 'Fx', 'G#', 'A', 'A#', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['B#', 'C#', 'D', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
            else:
                # print('Minor')
                pitch_name_temp = ['B#', 'C#', 'D', 'D#', 'E', 'E#', 'F#',
                                   'Fx' if '#11' in decided_chord_name else 'G', 'G#', 'A', 'A#', 'B']
        if decided_root_name == 'D':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['C', 'C#', 'D', 'Eb', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'C#', 'D', 'Eb', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'Cb']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G',
                                   'G#' if '#11' in decided_chord_name else 'Ab', 'A', 'Bb', 'B']
        if decided_root_name == 'Eb':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'Fb', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'Fb', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'Cb']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['Dbb', 'Db', 'D', 'Eb', 'Fb', 'F', 'Gb', 'G', 'Ab', 'Bbb', 'Bb', 'Cb']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'Fb', 'F', 'Gb', 'G', 'Ab',
                                   'A' if '#11' in decided_chord_name else 'Bbb', 'Bb', 'Cb']
        if decided_root_name == 'D#':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['B#', 'C#', 'Cx', 'D#', 'E', 'E#', 'Ex', 'Fx', 'G#', 'Gx', 'A#', 'Ax']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['B#', 'C#', 'Cx', 'D#', 'E', 'E#', 'Ex', 'Fx', 'G#', 'Gx', 'A#', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'C#', 'Cx', 'D#', 'E', 'E#', 'F#', 'Fx', 'G#', 'A', 'A#', 'B']
            else:
                # print('Minor')
                pitch_name_temp = ['B#', 'C#', 'Cx', 'D#', 'E', 'E#', 'F#', 'Fx', 'G#',
                                   'Gx' if '#11' in decided_chord_name else 'A', 'A#', 'B']
        if decided_root_name == 'E':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['B#', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'Fx', 'G#', 'A', 'A#', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'Fx', 'G#', 'A', 'A#', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'Db', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A',
                                   'A#' if '#11' in decided_chord_name else 'Bb', 'B']
        if decided_root_name == 'F':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'G#', 'A', 'Bb', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'G#', 'A', 'Bb', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'Db', 'Ebb', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb',
                                   'B' if '#11' in decided_chord_name else 'Cb']
        if decided_root_name == 'Gb':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'Fb', 'F', 'Gb', 'Abb', 'Ab', 'A', 'Bb', 'Cb']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'Db', 'Ebb', 'Eb', 'Fb', 'F', 'Gb', 'Abb', 'Ab', 'A', 'Bb', 'Cb']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['Dbb', 'Db', 'Ebb', 'Fbb', 'Fb', 'F', 'Gb', 'Abb', 'Ab', 'Bbb', 'Bb', 'Cb']
            else:
                # print('Minor')
                pitch_name_temp = ['C' if '#11' in decided_chord_name else 'Dbb',
                                   'Db', 'Ebb', 'Eb', 'Fb', 'F', 'Gb', 'Abb', 'Ab', 'Bbb', 'Bb', 'Cb']
        if decided_root_name == 'F#':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['B#', 'C#', 'Cx', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'Gx', 'A#', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['B#', 'C#', 'D', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'Gx', 'A#', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'C#', 'D', 'Eb', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            else:
                # print('Minor')
                pitch_name_temp = ['B#' if '#11' in decided_chord_name else 'C',
                                   'C#', 'D', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        if decided_root_name == 'G':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'A#', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'A#', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'Fb', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'C#' if '#11' in decided_chord_name else 'Db',
                                   'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        if decided_root_name == 'Ab':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'Bbb', 'Bb', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'Fb', 'F', 'Gb', 'G', 'Ab', 'Bbb', 'Bb', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'Db', 'Ebb', 'Eb', 'Fb', 'Gbb', 'Gb', 'G', 'Ab', 'Bbb', 'Bb', 'Cb']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'Db', 'D' if '#11' in decided_chord_name else 'Ebb',
                                   'Eb', 'Fb', 'F', 'Gb', 'G', 'Ab', 'Bbb', 'Bb', 'Cb']
        if decided_root_name == 'G#':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['B#', 'C#', 'Cx', 'D#', 'Dx', 'E#', 'F#', 'Fx', 'G#', 'A', 'A#', 'Ax']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['B#', 'C#', 'Cx', 'D#', 'E', 'E#', 'F#', 'Fx', 'G#', 'A', 'A#', 'Ax']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['B#', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'Fx', 'G#', 'A', 'A#', 'B']
            else:
                # print('Minor')
                pitch_name_temp = ['B#', 'C#', 'Cx' if '#11' in decided_chord_name else 'D',
                                   'D#', 'E', 'E#', 'F#', 'Fx', 'G#', 'A', 'A#', 'B']
        if decided_root_name == 'A':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['B#', 'C#', 'D', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['B#', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'G#', 'A', 'Bb', 'B']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'C#', 'D', 'D#' if '#11' in decided_chord_name else 'Eb',
                                   'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
        if decided_root_name == 'Bb':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'Cb']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'Fb', 'F', 'Gb', 'Abb', 'Ab', 'A', 'Bb', 'Cb']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'Db', 'D', 'Eb', 'E' if '#11' in decided_chord_name else 'Fb',
                                   'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'Cb']
        if decided_root_name == 'A#':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['B#', 'Bx', 'Cx', 'D#', 'Dx', 'E#', 'Ex', 'Fx', 'G#', 'Gx', 'A#', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['B#', 'Bx', 'Cx', 'D#', 'Dx', 'E#', 'F#', 'Fx', 'G#', 'Gx', 'A#', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['B#', 'C#', 'Cx', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'Gx', 'A#', 'B']
            else:
                # print('Minor')
                pitch_name_temp = ['B#', 'C#', 'Cx', 'D#', 'Dx' if '#11' in decided_chord_name else 'E',
                                   'E#', 'F#', 'Fx', 'G#', 'Gx', 'A#', 'B']
        if decided_root_name == 'B':
            if 'aug' in decided_chord_name or '#5' in decided_chord_name:
                # print('Augmented')
                pitch_name_temp = ['C', 'C#', 'Cx', 'D#', 'E', 'E#', 'F#', 'Fx', 'G#', 'A', 'A#', 'B']
            elif ('maj' in decided_chord_name) or ('m' not in decided_chord_name):
                # print('Major or Dominant')
                pitch_name_temp = ['C', 'C#', 'Cx', 'D#', 'E', 'E#', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            elif 'dim' in decided_chord_name:
                # print('Diminished')
                pitch_name_temp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'A#', 'B']
            else:
                # print('Minor')
                pitch_name_temp = ['C', 'C#', 'D', 'D#', 'E', 'E#' if '#11' in decided_chord_name else 'F',
                                   'F#', 'G', 'G#', 'A', 'A#', 'B']

    # 发生转位，规范低音标记
    if bass % 12 != decided_root % 12:
        # decided_res为完整的和弦判断结果（根音，低音，属性等等）
        decided_res += ('/' + pitch_name_temp[bass % 12])

    # 存入音名
    for sg_note in inp_dict:
        if 9 <= sg_note <= 11:
            # 大字二组
            if len(pitch_name_temp[sg_note]) >= 2:
                if pitch_name_temp[sg_note][1] == '#':
                    # 升
                    note_on_sheet.append([get_order(pitch_name_temp[sg_note][0]), 3, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note][1] == 'b':
                    if 'bb' in pitch_name_temp[sg_note]:
                        # 重降
                        adjust = 7 if pitch_name_temp[sg_note][0] == 'C' else 0
                        note_on_sheet.append([get_order(pitch_name_temp[sg_note][0]) + adjust, 1, inp_dict[sg_note]])
                    else:
                        # 降
                        adjust = 7 if pitch_name_temp[sg_note][0] == 'C' else 0
                        note_on_sheet.append([get_order(pitch_name_temp[sg_note][0]) + adjust, 2, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note][1] == 'x':
                    note_on_sheet.append([get_order(pitch_name_temp[sg_note][0]), 4, inp_dict[sg_note]])
            else:
                note_on_sheet.append([get_order(pitch_name_temp[sg_note][0]), 0, inp_dict[sg_note]])
        elif 12 <= sg_note <= 23:
            # 大字一组    
            if len(pitch_name_temp[sg_note % 12]) >= 2:
                if pitch_name_temp[sg_note % 12][1] == '#':
                    # 升
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 7 + adjust, 3, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'b':
                    if 'bb' in pitch_name_temp[sg_note % 12]:
                        # 重降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 7 + adjust, 1, inp_dict[sg_note]])
                    else:
                        # 降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 7 + adjust, 2, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'x':
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 7 + adjust, 4, inp_dict[sg_note]])
            else:
                note_on_sheet.append([get_order(pitch_name_temp[sg_note % 12][0]) + 7, 0, inp_dict[sg_note]])
        elif 24 <= sg_note <= 35:
            # 大字组
            if len(pitch_name_temp[sg_note % 12]) >= 2:
                if pitch_name_temp[sg_note % 12][1] == '#':
                    # 升
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 14 + adjust, 3, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'b':
                    if 'bb' in pitch_name_temp[sg_note % 12]:
                        # 重降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 14 + adjust, 1, inp_dict[sg_note]])
                    else:
                        # 降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 14 + adjust, 2, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'x':
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 14 + adjust, 4, inp_dict[sg_note]])
            else:
                note_on_sheet.append([get_order(pitch_name_temp[sg_note % 12][0]) + 14, 0, inp_dict[sg_note]])
        elif 36 <= sg_note <= 47:
            # 小字组
            if len(pitch_name_temp[sg_note % 12]) >= 2:
                if pitch_name_temp[sg_note % 12][1] == '#':
                    # 升
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 21 + adjust, 3, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'b':
                    if 'bb' in pitch_name_temp[sg_note % 12]:
                        # 重降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 21 + adjust, 1, inp_dict[sg_note]])
                    else:
                        # 降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 21 + adjust, 2, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'x':
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 21 + adjust, 4, inp_dict[sg_note]])
            else:
                note_on_sheet.append([get_order(pitch_name_temp[sg_note % 12][0]) + 21, 0, inp_dict[sg_note]])
        elif 48 <= sg_note <= 59:
            # 小字一组
            if len(pitch_name_temp[sg_note % 12]) >= 2:
                if pitch_name_temp[sg_note % 12][1] == '#':
                    # 升
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 28 + adjust, 3, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'b':
                    if 'bb' in pitch_name_temp[sg_note % 12]:
                        # 重降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 28 + adjust, 1, inp_dict[sg_note]])
                    else:
                        # 降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 28 + adjust, 2, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'x':
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 28 + adjust, 4, inp_dict[sg_note]])
            else:
                note_on_sheet.append([get_order(pitch_name_temp[sg_note % 12][0]) + 28, 0, inp_dict[sg_note]])
        elif 60 <= sg_note <= 71:
            # 小字二组
            if len(pitch_name_temp[sg_note % 12]) >= 2:
                if pitch_name_temp[sg_note % 12][1] == '#':
                    # 升
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 35 + adjust, 3, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'b':
                    if 'bb' in pitch_name_temp[sg_note % 12]:
                        # 重降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 35 + adjust, 1, inp_dict[sg_note]])
                    else:
                        # 降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 35 + adjust, 2, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'x':
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 35 + adjust, 4, inp_dict[sg_note]])
            else:
                note_on_sheet.append([get_order(pitch_name_temp[sg_note % 12][0]) + 35, 0, inp_dict[sg_note]])
        elif 72 <= sg_note <= 83:
            # 小字三组
            if len(pitch_name_temp[sg_note % 12]) >= 2:
                if pitch_name_temp[sg_note % 12][1] == '#':
                    # 升
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 42 + adjust, 3, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'b':
                    if 'bb' in pitch_name_temp[sg_note % 12]:
                        # 重降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 42 + adjust, 1, inp_dict[sg_note]])
                    else:
                        # 降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 42 + adjust, 2, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'x':
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 42 + adjust, 4, inp_dict[sg_note]])
            else:
                note_on_sheet.append([get_order(pitch_name_temp[sg_note % 12][0]) + 42, 0, inp_dict[sg_note]])
        elif 84 <= sg_note <= 95:
            # 小字四组
            if len(pitch_name_temp[sg_note % 12]) >= 2:
                if pitch_name_temp[sg_note % 12][1] == '#':
                    # 升
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 49 + adjust, 3, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'b':
                    if 'bb' in pitch_name_temp[sg_note % 12]:
                        # 重降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 49 + adjust, 1, inp_dict[sg_note]])
                    else:
                        # 降
                        adjust = 7 if pitch_name_temp[sg_note % 12][0] == 'C' else 0
                        note_on_sheet.append(
                            [get_order(pitch_name_temp[sg_note % 12][0]) + 49 + adjust, 2, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'x':
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 49 + adjust, 4, inp_dict[sg_note]])
            else:
                note_on_sheet.append([get_order(pitch_name_temp[sg_note % 12][0]) + 49, 0, inp_dict[sg_note]])
        elif sg_note == 96:
            # 最高音C5
            if len(pitch_name_temp[sg_note % 12]) >= 2:
                if pitch_name_temp[sg_note % 12][1] == '#':
                    # 升
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 56 + adjust, 3, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'b':
                    if 'bb' in pitch_name_temp[sg_note % 12]:
                        # 重降
                        note_on_sheet.append([get_order(pitch_name_temp[sg_note % 12][0]) + 56, 1, inp_dict[sg_note]])
                    else:
                        # 降
                        note_on_sheet.append([get_order(pitch_name_temp[sg_note % 12][0]) + 56, 2, inp_dict[sg_note]])
                elif pitch_name_temp[sg_note % 12][1] == 'x':
                    adjust = -7 if pitch_name_temp[sg_note % 12][0] == 'B' else 0
                    note_on_sheet.append(
                        [get_order(pitch_name_temp[sg_note % 12][0]) + 56 + adjust, 4, inp_dict[sg_note]])
            else:
                note_on_sheet.append([get_order(pitch_name_temp[sg_note % 12][0]) + 56, 0, inp_dict[sg_note]])

    return decided_res, note_on_sheet
