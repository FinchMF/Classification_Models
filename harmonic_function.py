from collections import Counter

music_m = [ {'Major': [
    {
        'key':'C',
        'C':'I',
        'C#/Db':'bii',
        'D':'ii',
        'D#/Eb':'biii',
        'E':'iii',
        'F':'IV',
        'F#/Gb':'#iv',
        'G':'V',
        'G#/Ab':'bvi',
        'A':'vi',
        'A#/Bb':'bvii',
        'B':'vii/o'
    },
    {
        'key':'C#/Db',
        'C':'vii/o',
        'C#/Db':'I',
        'D':'bii',
        'D#/Eb':'ii',
        'E':'biii',
        'F':'iii',
        'F#/Gb':'IV',
        'G':'#iv',
        'G#/Ab':'V',
        'A':'bvi',
        'A#/Bb':'vi',
        'B':'bvii'
    },
    {
        'key':'D',
        'C':'bvii',
        'C#/Db':'vii/o',
        'D':'I',
        'D#/Eb':'bii',
        'E':'ii',
        'F':'biii',
        'F#/Gb':'iii',
        'G':'IV',
        'G#/Ab':'#iv',
        'A':'V',
        'A#/Bb':'bvi',
        'B':'vi'
    },
    {
        'key':'D#/Eb',
        'C':'vi',
        'C#/Db':'bvii',
        'D':'vii/o',
        'D#/Eb':'I',
        'E':'bii',
        'F':'ii',
        'F#/Gb':'biii',
        'G':'iii',
        'G#/Ab':'IV',
        'A':'#iv',
        'A#/Bb':'V',
        'B':'bvi'
    },
    {
        'key':'E',
        'C':'bvi',
        'C#/Db':'vi',
        'D':'bvii',
        'D#/Eb':'vii/o',
        'E':'I',
        'F':'bii',
        'F#/Gb':'ii',
        'G':'biii',
        'G#/Ab':'iii',
        'A':'IV',
        'A#/Bb':'#iv',
        'B':'V'
    },
    {
        'key':'F',
        'C':'V',
        'C#/Db':'bvi',
        'D':'vi',
        'D#/Eb':'bvii',
        'E':'vii/o',
        'F':'I',
        'F#/Gb':'bii',
        'G':'ii',
        'G#/Ab':'biii',
        'A':'iii',
        'A#/Bb':'IV',
        'B':'#iv'
    },
    {
        'key':'F#/Gb',
        'C':'#iv',
        'C#/Db':'V',
        'D':'bvi',
        'D#/Eb':'vi',
        'E':'bvii',
        'F':'vii/o',
        'F#/Gb':'I',
        'G':'bii',
        'G#/Ab':'ii',
        'A':'biii',
        'A#/Bb':'iii',
        'B':'IV'
    },
    {
        'key':'G',
        'C':'IV',
        'C#/Db':'#iv',
        'D':'V',
        'D#/Eb':'bvi',
        'E':'vi',
        'F':'bvii',
        'F#/Gb':'vii/o',
        'G':'I',
        'G#/Ab':'bii',
        'A':'ii',
        'A#/Bb':'biii',
        'B':'iii'
    },
    {
        'key':'G#/Ab',
        'C':'iii',
        'C#/Db':'IV',
        'D':'#iv',
        'D#/Eb':'V',
        'E':'bvi',
        'F':'vi',
        'F#/Gb':'bvii',
        'G':'vii/o',
        'G#/Ab':'I',
        'A':'bii',
        'A#/Bb':'ii',
        'B':'biii'
    },
    {
        'key':'A',
        'C':'biii',
        'C#/Db':'iii',
        'D':'IV',
        'D#/Eb':'#iv',
        'E':'V',
        'F':'bvi',
        'F#/Gb':'vi',
        'G':'bvii',
        'G#/Ab':'vii/o',
        'A':'I',
        'A#/Bb':'bii',
        'B':'ii'
    },
    {
        'key':'A#/Bb',
        'C':'ii',
        'C#/Db':'biii',
        'D':'iii',
        'D#/Eb':'IV',
        'E':'#iv',
        'F':'V',
        'F#/Gb':'bvi',
        'G':'vi',
        'G#/Ab':'bvii',
        'A':'vii/o',
        'A#/Bb':'I',
        'B':'bii'
    },
    {
        'key':'B',
        'C':'bii',
        'C#/Db':'ii',
        'D':'biii',
        'D#/Eb':'iii',
        'E':'IV',
        'F':'#iv',
        'F#/Gb':'V',
        'G':'bvi',
        'G#/Ab':'vi',
        'A':'bvii',
        'A#/Bb':'vii/o',
        'B':'I'
    }]},
         {'Minor': [
    {
        'key':'C',
        'C':'i',
        'C#/Db':'bii',
        'D':'ii/o',
        'D#/Eb':'bIII',
        'E':'iii',
        'F':'iv',
        'F#/Gb':'bv',
        'G':'v',
        'G#/Ab':'bVI',
        'A':'vi',
        'A#/Bb':'bVII',
        'B':'vii'
    },
    {
        'key':'C#/Db',
        'C':'vii',
        'C#/Db':'i',
        'D':'bii',
        'D#/Eb':'ii/o',
        'E':'bIII',
        'F':'iii',
        'F#/Gb':'iv',
        'G':'bv',
        'G#/Ab':'v',
        'A':'bVI',
        'A#/Bb':'vi',
        'B':'bVII'
    },
    {
        'key':'D',
        'C':'bVII',
        'C#/Db':'vii',
        'D':'i',
        'D#/Eb':'bii',
        'E':'ii/o',
        'F':'bIII',
        'F#/Gb':'iii',
        'G':'iv',
        'G#/Ab':'bv',
        'A':'v',
        'A#/Bb':'bVI',
        'B':'vi'
    },
    {
        'key':'D#/Eb',
        'C':'vi',
        'C#/Db':'bVII',
        'D':'vii',
        'D#/Eb':'i',
        'E':'bii',
        'F':'ii/o',
        'F#/Gb':'bIII',
        'G':'iii',
        'G#/Ab':'iv',
        'A':'bv',
        'A#/Bb':'v',
        'B':'bVI'
    },
    {
        'key':'E',
        'C':'bVI',
        'C#/Db':'vi',
        'D':'bVII',
        'D#/Eb':'vii',
        'E':'i',
        'F':'bii',
        'F#/Gb':'ii/o',
        'G':'bIII',
        'G#/Ab':'iii',
        'A':'iv',
        'A#/Bb':'bv',
        'B':'v'
    },
    {
        'key':'F',
        'C':'v',
        'C#/Db':'bVI',
        'D':'vi',
        'D#/Eb':'bVII',
        'E':'vii',
        'F':'i',
        'F#/Gb':'bii',
        'G':'ii/o',
        'G#/Ab':'bIII',
        'A':'iii',
        'A#/Bb':'iv',
        'B':'bv'
    },
    {
        'key':'F#/Gb',
        'C':'bv',
        'C#/Db':'v',
        'D':'bVI',
        'D#/Eb':'vi',
        'E':'bvii',
        'F':'vii/o',
        'F#/Gb':'i',
        'G':'bii',
        'G#/Ab':'ii/o',
        'A':'bIII',
        'A#/Bb':'iii',
        'B':'iv'
    },
    {
        'key':'G',
        'C':'iv',
        'C#/Db':'bv',
        'D':'v',
        'D#/Eb':'bVI',
        'E':'vi',
        'F':'bVII',
        'F#/Gb':'vii',
        'G':'i',
        'G#/Ab':'bii',
        'A':'ii/o',
        'A#/Bb':'bIII',
        'B':'iii'
    },
    {
        'key':'G#/Ab',
        'C':'iii',
        'C#/Db':'iv',
        'D':'bv',
        'D#/Eb':'v',
        'E':'bVI',
        'F':'vi',
        'F#/Gb':'bVII',
        'G':'vii',
        'G#/Ab':'i',
        'A':'bii',
        'A#/Bb':'ii/o',
        'B':'bIII'
    },
    {
        'key':'A',
        'C':'bIII',
        'C#/Db':'iii',
        'D':'iv',
        'D#/Eb':'bv',
        'E':'v',
        'F':'bVI',
        'F#/Gb':'vi',
        'G':'bVII',
        'G#/Ab':'vii',
        'A':'i',
        'A#/Bb':'bii',
        'B':'ii/o'
    },
    {
        'key':'A#/Bb',
        'C':'ii/o',
        'C#/Db':'bIII',
        'D':'iii',
        'D#/Eb':'iv',
        'E':'bv',
        'F':'v',
        'F#/Gb':'bVI',
        'G':'vi',
        'G#/Ab':'bVII',
        'A':'vii',
        'A#/Bb':'i',
        'B':'bii'
    },
    {
        'key':'B',
        'C':'bii',
        'C#/Db':'ii/o',
        'D':'bIII',
        'D#/Eb':'iii',
        'E':'iv',
        'F':'bv',
        'F#/Gb':'V',
        'G':'bVI',
        'G#/Ab':'vi',
        'A':'bVII',
        'A#/Bb':'vii',
        'B':'i'
    }]}

]

# parses three lists, mode, musical_key and progression to return roman numerial analysis
def get_progression_m(mode, musical_key, progression):
    j = 0
    return_progression = []
    while j < len(progression):
        if mode == 'Major':
            for i in range(len(music_m[0]['Major'])):
                if music_m[0]['Major'][i]['key'] == musical_key:
                    return_progression.append(music_m[0]['Major'][i][progression[j]])
                else:
                    continue
        else:
            for i in range(len(music_m[1]['Minor'])):
                if music_m[1]['Minor'][i]['key'] == musical_key:
                    return_progression.append(music_m[1]['Minor'][i][progression[j]])
                else:
                    continue
        j +=1
    return return_progression



# function that takes the abstracted roman numeral analysis of spotify audio analysis 
# and returns a count each of harmony in the list of tracks

# first the function takes in the exact result of harmonic segment by number of segments
def get_reduced_abstraction(data):
    reduced_abst_prog = []
    for i in (data):
# here the roman numerial analysis is reduced to one expression of the chord
        reduce = []
        for x in i:
            if x not in reduce:
                reduce.append(x)
# the reduced figure is returned to the list in the order of the tracks
        reduced_abst_prog.append(reduce)
# sorting the reduced lists of roman numeral analysis to show where harmonic form is similar through between tracks
        for i in reduced_abst_prog:
            i.sort()
# making a dictionary
    sort_harmony = dict(zip(list(range(0,len(reduced_abst_prog))),reduced_abst_prog))
# turning the values of the dictionary into tuples
    s_harmony = []
    for k,v in sort_harmony.items():
        s_harmony.append(tuple(v))
# counting the tuples and returning them as a dictionary of counts
        sorts = Counter(s_harmony)
    return sorts