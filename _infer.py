import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import AutoTokenizer, get_linear_schedule_with_warmup, modeling_gpt2
import OpenMatch as om

from tqdm import tqdm
import time
import os

def dev(args, model, metric, dev_loader, device):
    rst_dict = {}
    for dev_batch in tqdm(dev_loader):
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], \
                                                   dev_batch['retrieval_score']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device),
                                       dev_batch['segment_ids'].to(device))
            elif args.model == 'roberta':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device))
            elif args.model == 'edrm':
                batch_score, _ = model(dev_batch['query_wrd_idx'].to(device), dev_batch['query_wrd_mask'].to(device),
                                       dev_batch['doc_wrd_idx'].to(device), dev_batch['doc_wrd_mask'].to(device),
                                       dev_batch['query_ent_idx'].to(device), dev_batch['query_ent_mask'].to(device),
                                       dev_batch['doc_ent_idx'].to(device), dev_batch['doc_ent_mask'].to(device),
                                       dev_batch['query_des_idx'].to(device), dev_batch['doc_des_idx'].to(device))
            else:
                batch_score, _ = model(dev_batch['query_idx'].to(device), dev_batch['query_mask'].to(device),
                                       dev_batch['doc_idx'].to(device), dev_batch['doc_mask'].to(device))
            if args.task == 'classification':
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id in rst_dict:
                    rst_dict[q_id].append((b_s, d_id, l))
                else:
                    rst_dict[q_id] = [(b_s, d_id, l)]
    return rst_dict


def train_reinfoselect(args, model, policy, loss_fn, m_optim, m_scheduler, p_optim, metric, train_loader, dev_loader,
                       device):
    best_mes = 0.0
    with torch.no_grad():
        rst_dict = dev(args, model, metric, dev_loader, device)
        om.utils.save_trec(args.res, rst_dict)
        if args.metric.split('_')[0] == 'mrr':
            mes = metric.get_mrr(args.qrels, args.res, args.metric)
        else:
            mes = metric.get_metric(args.qrels, args.res, args.metric)
    if mes >= best_mes:
        best_mes = mes
        print('save_model...')
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), args.save)
        else:
            torch.save(model.state_dict(), args.save)
    print('initial result: ', mes)
    last_mes = mes
    for epoch in range(args.epoch):
        avg_loss = 0.0
        log_prob_ps = []
        log_prob_ns = []
        for step, train_batch in enumerate(train_loader):
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['input_ids_pos'].to(device),
                                            train_batch['input_mask_pos'].to(device),
                                            train_batch['segment_ids_pos'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device),
                                            train_batch['segment_ids'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['input_ids_pos'].to(device),
                                            train_batch['input_mask_pos'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['query_wrd_idx'].to(device),
                                            train_batch['query_wrd_mask'].to(device),
                                            train_batch['doc_pos_wrd_idx'].to(device),
                                            train_batch['doc_pos_wrd_mask'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['query_wrd_idx'].to(device),
                                            train_batch['query_wrd_mask'].to(device),
                                            train_batch['doc_wrd_idx'].to(device),
                                            train_batch['doc_wrd_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                            train_batch['doc_pos_idx'].to(device),
                                            train_batch['doc_pos_mask'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                            train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            batch_probs = F.gumbel_softmax(batch_probs, tau=args.tau)
            m = Categorical(batch_probs)
            action = m.sample()
            if action.sum().item() < 1:
                m_scheduler.step()
                if (step + 1) % args.eval_every == 0 and len(log_prob_ps) > 0:
                    with torch.no_grad():
                        rst_dict = dev(args, model, metric, dev_loader, device)
                        om.utils.save_trec(args.res, rst_dict)
                        if args.metric.split('_')[0] == 'mrr':
                            mes = metric.get_mrr(args.qrels, args.res, args.metric)
                        else:
                            mes = metric.get_metric(args.qrels, args.res, args.metric)
                    if mes >= best_mes:
                        best_mes = mes
                        print('save_model...')
                        if torch.cuda.device_count() > 1:
                            torch.save(model.module.state_dict(), args.save)
                        else:
                            torch.save(model.state_dict(), args.save)
                    print(step + 1, avg_loss / len(log_prob_ps), mes, best_mes)
                    avg_loss = 0.0

                    reward = mes - last_mes
                    last_mes = mes
                    if reward >= 0:
                        policy_loss = [(-log_prob_p * reward).sum().unsqueeze(-1) for log_prob_p in log_prob_ps]
                    else:
                        policy_loss = [(log_prob_n * reward).sum().unsqueeze(-1) for log_prob_n in log_prob_ns]
                    policy_loss = torch.cat(policy_loss).sum()
                    policy_loss.backward()
                    p_optim.step()
                    p_optim.zero_grad()

                    if args.reset:
                        state_dict = torch.load(args.save)
                        model.load_state_dict(state_dict)
                        last_mes = best_mes
                    log_prob_ps = []
                    log_prob_ns = []
                continue

            filt = action.nonzero().squeeze(-1).cpu()
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].index_select(0, filt).to(device),
                                               train_batch['input_mask_pos'].index_select(0, filt).to(device),
                                               train_batch['segment_ids_pos'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].index_select(0, filt).to(device),
                                               train_batch['input_mask_neg'].index_select(0, filt).to(device),
                                               train_batch['segment_ids_neg'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].index_select(0, filt).to(device),
                                           train_batch['input_mask'].index_select(0, filt).to(device),
                                           train_batch['segment_ids'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].index_select(0, filt).to(device),
                                               train_batch['input_mask_pos'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].index_select(0, filt).to(device),
                                               train_batch['input_mask_neg'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].index_select(0, filt).to(device),
                                           train_batch['input_mask'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device),
                                               train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_wrd_idx'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['query_ent_idx'].index_select(0, filt).to(device),
                                               train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_ent_idx'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['query_des_idx'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_des_idx'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device),
                                               train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_wrd_idx'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['query_ent_idx'].index_select(0, filt).to(device),
                                               train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_ent_idx'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['query_des_idx'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_des_idx'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device),
                                           train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_wrd_idx'].index_select(0, filt).to(device),
                                           train_batch['doc_wrd_mask'].index_select(0, filt).to(device),
                                           train_batch['query_ent_idx'].index_select(0, filt).to(device),
                                           train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_ent_idx'].index_select(0, filt).to(device),
                                           train_batch['doc_ent_mask'].index_select(0, filt).to(device),
                                           train_batch['query_des_idx'].index_select(0, filt).to(device),
                                           train_batch['doc_des_idx'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_idx'].index_select(0, filt).to(device),
                                               train_batch['query_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_idx'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_mask'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['query_idx'].index_select(0, filt).to(device),
                                               train_batch['query_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_idx'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_mask'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_idx'].index_select(0, filt).to(device),
                                           train_batch['query_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_idx'].index_select(0, filt).to(device),
                                           train_batch['doc_mask'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')

            log_prob_p = m.log_prob(action)
            log_prob_n = m.log_prob(1 - action)
            log_prob_ps.append(log_prob_p)
            log_prob_ns.append(log_prob_n)

            if args.task == 'ranking':
                batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(),
                                     torch.ones(batch_score_pos.size()).to(device))
            elif args.task == 'classification':
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
            if torch.cuda.device_count() > 1:
                batch_loss = batch_loss.mean(-1)
            batch_loss = batch_loss.mean()
            avg_loss += batch_loss.item()
            batch_loss.backward()
            m_optim.step()
            m_scheduler.step()
            m_optim.zero_grad()

            if (step + 1) % args.eval_every == 0:
                with torch.no_grad():
                    rst_dict = dev(args, model, metric, dev_loader, device)
                    om.utils.save_trec(args.res, rst_dict)
                    if args.metric.split('_')[0] == 'mrr':
                        mes = metric.get_mrr(args.qrels, args.res, args.metric)
                    else:
                        mes = metric.get_metric(args.qrels, args.res, args.metric)
                if mes >= best_mes:
                    best_mes = mes
                    print('save_model...')
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), args.save)
                    else:
                        torch.save(model.state_dict(), args.save)
                print(step + 1, avg_loss / len(log_prob_ps), mes, best_mes)
                avg_loss = 0.0

                reward = mes - last_mes
                last_mes = mes
                if reward >= 0:
                    policy_loss = [(-log_prob_p * reward).sum().unsqueeze(-1) for log_prob_p in log_prob_ps]
                else:
                    policy_loss = [(log_prob_n * reward).sum().unsqueeze(-1) for log_prob_n in log_prob_ns]
                policy_loss = torch.cat(policy_loss).sum()
                policy_loss.backward()
                p_optim.step()
                p_optim.zero_grad()

                if args.reset:
                    state_dict = torch.load(args.save)
                    model.load_state_dict(state_dict)
                    last_mes = best_mes
                log_prob_ps = []
                log_prob_ns = []


def infer(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, dev_loader_2, device, writer,
          teacher_model=None, loss_fn_2=None):


    model.eval()
    with torch.no_grad():
        print("running inference...")


        rst_dict = dev(args, model, metric, dev_loader, device)
        om.utils.save_trec(args.res, rst_dict)
        mes = metric.get_mrr(args.qrels, args.res, args.metric)
        writer.add_scalar("dev", mes, step)
        print("Step " + str(step) + ": " + str(args.metric) + " = " + str(mes))





def make_task_work_dir(args):
    if args.task == 'new':
        if not args.mse and not args.ranking_loss:
            args.ranking = True
            print('task new setted to be rank only!')

    # make task_work_dir
    time_prefix = time.strftime("%m%d_%H%M%S", time.localtime())
    task_str = 'task_{task}_mse_{mse}_rank_{rank}'.format(task=args.task, mse=str(args.mse), rank=str(args.ranking_loss))
    task_work_dir = "{dir}/{time}/{task}".format(dir=args.log_dir, time=time_prefix, task=task_str)


    args.log_dir = task_work_dir + '/tensorboard_logs'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # make result dir
    result_dir = "{dir}/{time}/{task}".format(dir=args.res, time=time_prefix, task=task_str)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    checkpoint_dir = task_work_dir + '/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args.save = checkpoint_dir + '/cpt_'

    args.res = task_work_dir + '/dev.trec'
    args.res2 = task_work_dir + '/test.trec'


    # command log
    argsDict = args.__dict__
    with open(task_work_dir + '/command.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mse', action='store_true', default=False)
    parser.add_argument('-ranking_loss', action='store_true', default=False)
    parser.add_argument('-eval_first', action='store_true', default=False)

    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-reinfoselect', action='store_true', default=False)
    parser.add_argument('-reset', action='store_true', default=False)
    parser.add_argument('-train', action=om.utils.DictOrStr, default='./data/train_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-dev', action=om.utils.DictOrStr)
    parser.add_argument('-dev2', action=om.utils.DictOrStr)
    parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    parser.add_argument("-qrels2", type=str)
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument("-res2", type=str)
    parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_doc_len', type=int, default=150)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-eval_every', type=int, default=1000)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--teach", action="store_true")

    args = parser.parse_args()

    make_task_work_dir(args)



    writer = SummaryWriter(args.log_dir)

    args.model = args.model.lower()
    dev_set_2 = None
    dev_set = None
    # Preparing Dataset.
    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        train_set = om.data.datasets.BertDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        if args.dev != None:
            dev_set = om.data.datasets.BertDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        print('reading test data...')
        if args.dev2 != None:
            dev_set_2 = om.data.datasets.BertDataset(
                dataset=args.dev2,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
    elif args.model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        train_set = om.data.datasets.RobertaDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.RobertaDataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
    elif args.model == 'edrm':
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        ent_tokenizer = om.data.tokenizers.WordTokenizer(
            vocab=args.ent_vocab
        )
        print('reading training data...')
        train_set = om.data.datasets.EDRMDataset(
            dataset=args.train,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.EDRMDataset(
            dataset=args.dev,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
    else:
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        print('reading training data...')
        train_set = om.data.datasets.Dataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.Dataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )

    train_loader = None
    dev_loader, dev_loader_2 = None, None
    if dev_set != None:
        dev_loader = om.data.DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size * 16,
            shuffle=False,
            num_workers=8
        )
    if dev_set_2 != None:
        dev_loader_2 = om.data.DataLoader(
            dataset=dev_set_2,
            batch_size=args.batch_size * 16,
            shuffle=False,
            num_workers=8
        )

    # Preparing Model.
    model, model2 = None, None
    if args.model == 'bert' or args.model == 'roberta':
        model = om.models.Bert(
            pretrained=args.pretrain,
            mode=args.mode,
            task=args.task
        )
        if args.teach:
            model2 = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task=args.task
            )
        if args.reinfoselect:
            policy = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task='classification'
            )
    elif args.model == 'edrm':
        model = om.models.EDRM(
            wrd_vocab_size=tokenizer.get_vocab_size(),
            ent_vocab_size=ent_tokenizer.get_vocab_size(),
            wrd_embed_dim=tokenizer.get_embed_dim(),
            ent_embed_dim=128,
            max_des_len=20,
            max_ent_num=3,
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            wrd_embed_matrix=tokenizer.get_embed_matrix(),
            ent_embed_matrix=None,
            task=args.task
        )
    elif args.model == 'tk':
        model = om.models.TK(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            head_num=10,
            hidden_dim=100,
            layer_num=2,
            kernel_num=args.n_kernels,
            dropout=0.0,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model == 'cknrm':
        model = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model == 'knrm':
        model = om.models.KNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    else:
        raise ValueError('model name error.')

    if args.reinfoselect and args.model != 'bert':
        policy = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task='classification'
        )

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)
        if args.model == 'bert':
            st = {}
            for k in state_dict:
                if k.startswith('bert'):
                    st['_model' + k[len('bert'):]] = state_dict[k]
                elif k.startswith('classifier'):
                    st['_dense' + k[len('classifier'):]] = state_dict[k]
                else:
                    st[k] = state_dict[k]
            model.load_state_dict(st)
            if model2 != None:
                model2.load_state_dict(st)
        else:
            model.load_state_dict(state_dict)
            if model2 != None:
                model2.load_state_dict(state_dict)

    # debug
    # device = torch.device('cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loss function
    loss_fn, loss_fn_2 = None, None
    if args.reinfoselect:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
        elif args.task == 'new':
            loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
            loss_fn_2 = nn.MSELoss()
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
    else:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1)
        elif args.task == 'new':
            loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
            loss_fn_2 = nn.MSELoss()
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
    m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps,
                                                  num_training_steps=len(train_set) * args.epoch // args.batch_size)
    if args.reinfoselect:
        p_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr)
    metric = om.metrics.Metric()

    model.to(device)
    if model2 != None:
        model2.to(device)
    if args.reinfoselect:
        policy.to(device)
    loss_fn.to(device) 
    if loss_fn_2 != None:
        loss_fn_2.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if model2 != None:
            model2 = nn.DataParallel(model2)
        loss_fn = nn.DataParallel(loss_fn)
        if loss_fn_2 != None:
            loss_fn_2 = nn.DataParallel(loss_fn_2)

    if args.reinfoselect:
        train_reinfoselect(args, model, policy, loss_fn, m_optim, m_scheduler, p_optim, metric, train_loader,
                           dev_loader, device)
    else:
        infer(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, dev_loader_2, device,
              writer, teacher_model=model2, loss_fn_2=loss_fn_2)
    writer.close()

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), args.save)
    else:
        torch.save(model.state_dict(), args.save)


if __name__ == "__main__":
    main()
