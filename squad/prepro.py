import argparse
import json
import os
from nltk.parse.stanford import StanfordParser, StanfordDependencyParser

# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm

from squad.utils import get_word_span, get_word_idx, process_tokens


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/squad"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument("--train_name", default='train-v1.1.json')
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    parser.add_argument("--suffix", default="")
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, args.train_name)
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, args.dev_name)
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.mode == 'full':
        prepro_each(args, 'train', out_name='train')
        # prepro_each(args, 'dev', out_name='dev')
        # prepro_each(args, 'dev', out_name='test')
    elif args.mode == 'all':
        create_all(args)
        prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
        prepro_each(args, 'dev', 0.0, 0.0, out_name='test')
        prepro_each(args, 'all', out_name='train')
    elif args.mode == 'single':
        assert len(args.single_path) > 0
        prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    else:
        prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
        prepro_each(args, 'train', args.train_ratio, 1.0, out_name='dev')
        prepro_each(args, 'dev', out_name='test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}_1.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}_1.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter),
                                                                        glove_path))
    return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    parser = StanfordParser(model_path=os.getenv("StanfordParser_model_path"))
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize

        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    # if not args.split:
    #     sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "{}-{}v1.1.json".format(data_type, args.suffix))
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    q_syn_seq = []
    na = []
    cy = []
    x, cx = [], []
    syn_seq = []
    rsyn_seq = []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    syn_counter = Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    abandon = 0
    for ai, article in enumerate(tqdm(source_data['data'][221:])):
        xp, cxp = [], []
        syn_seqp = []
        pp = []
        x.append(xp)
        cx.append(cxp)
        syn_seq.append(syn_seqp)
        p.append(pp)
        p_i = -1
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ').replace("``", '" ').replace('e.g.', 'e-g,')
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            try:
                syn_seqpp = sents_parser(xi, parser)
                p_i += 1
            except:
                abandon += 1
                continue
            for sent in syn_seqpp:
                for word in sent:
                    for syn in word:
                        syn_counter[syn] += 1

            syn_seqp.append(syn_seqpp)
            # given xi, add chars
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai+221, p_i]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == p_i
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                qi = process_tokens(qi)
                try:
                    q_syn_seqq = sent_parser(qi, parser)
                except:
                    continue
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    # TODO : put some function that gives word_start, word_stop here
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    w0 = xi[yi0[0]][yi0[1]]
                    w1 = xi[yi1[0]][yi1[1] - 1]
                    i0 = get_word_idx(context, xi, yi0)
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1] - 1))
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])

                if len(qa['answers']) == 0:
                    yi.append([(0, 0), (0, 1)])
                    cyi.append([0, 1])
                    na.append(True)
                else:
                    na.append(False)

                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1


                q.append(qi)
                cq.append(cqi)
                q_syn_seq.append(q_syn_seqq)
                y.append(yi)
                cy.append(cyi)
                rx.append(rxi)
                rcx.append(rxi)
                rsyn_seq.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                answerss.append(answers)
        print('abandon {} paragraph'.format(abandon))
        if args.debug:
            break
    for sent in q_syn_seq:
        for word in sent:
            for syn in word:
                syn_counter[syn] += 1
    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, '*syn_seq': rsyn_seq, 'cy': cy, 'q_syn_seq': q_syn_seq,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx, 'na': na}
    shared = {'x': x, 'cx': cx, 'p': p, 'syn_seq': syn_seq, 'syn_counter': syn_counter,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}
    print("saving ...")
    save(args, data, shared, out_name)


def sents_parser(sent, parser):
    sents = parser.parse_sents(sent)
    syn_seqss = []
    for sent in sents:
        syn_seqs = []
        sent = next(sent)
        for leaf in range(len(sent.leaves())):
            syn_seq = []
            path = sent.leaf_treeposition(leaf)
            for i in range(10):
                syn_seq.append(sent[path[:-1 - i]].label())
            syn_seqs.append(syn_seq)
        syn_seqss.append(syn_seqs)
    return syn_seqss


def sent_parser(sent, parser):
    sent = parser.parse(sent)
    syn_seqs = []
    sent = next(sent)
    for leaf in range(len(sent.leaves())):
        syn_seq = []
        path = sent.leaf_treeposition(leaf)
        for i in range(10):
            syn_seq.append(sent[path[:-1 - i]].label())
        syn_seqs.append(syn_seq)
    return syn_seqs


if __name__ == "__main__":
    main()
