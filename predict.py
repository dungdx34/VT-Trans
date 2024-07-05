from fairseq.models.transformer import TransformerModel
import re

en2vi = TransformerModel.from_pretrained(
        'exp_mbart_doc/run-mbart/iwslt.checkpoints.en-vi',
        task='translation_doc',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='exp_mbart_doc/iwslt.binarized.en-vi/',
        bpe='sentencepiece',
        sentencepiece_vocab='/storage-nlp/nlp/dungdx4/BERT/mbart.cc25/sentence.bpe.model'
        )
#sent =  en2vi.translate('Hello world', beam=5)
#print(sent)
en2vi.cuda()

def clean_seq(string):
    string = re.sub(r" ,", ",",string)
    string = re.sub(r" \.", ".",string)
    string = re.sub(r" !", "!",string)
    string = re.sub(r" \?", "?",string)
    string = re.sub(r" \(", "(",string)
    string = re.sub(r" \)", ")",string)
    string = re.sub(r"\s{2,}", " ",string)
    return string.strip()

with open('test/test-iwslt-doc-en.txt', 'w') as fout:
    with open('raw_data/iwslt/doc-pair/test.en', 'r') as fin:
        line = fin.readline()
        while line != '':
            line = line.strip()
            print(line)
            try:
                #sent =  en2vi.translate(line, beam=5, skip_invalid_size_inputs=True)
                sent =  en2vi.translate(line, beam=5)
            except:
                count = int(len(line)/900)
                index = 0
                sent = ''
                while index < count:
                    index += 1
                    sent1 = line[(index-1)*900:index*900]
                    sent += en2vi.translate(sent1, beam=5) + ' '
                sent = sent.strip()
                pass
            #sent = clean_seq(sent)
            print(sent)
            fout.write(sent + '\n')
            line = fin.readline()

