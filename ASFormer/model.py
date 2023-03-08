import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
import copy
import numpy as np
import math
from tqdm import tqdm
from eval import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p * idx_decoder)


class AttentionHelper(nn.Module):
    def __init__(self, future_window):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.future_window = future_window

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape

        assert c1 == c2

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        bounded_future_mask = torch.tril(padding_mask, diagonal=l1 // 2 + self.future_window)
        energy *= bounded_future_mask
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(bounded_future_mask + 1e-6)  # log(1e-6) for zero paddings
        # attention = attention + torch.log(padding_mask + 1e-6)
        attention = self.softmax(attention)
        # attention = attention * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, future_window):  # r1 = r2
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)

        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder', 'decoder']

        self.att_helper = AttentionHelper(future_window=future_window)
        self.window_mask = self.construct_window_mask()

    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2 * (self.bl // 2)))
        for i in range(self.bl):
            window_mask[:, :, i:i + self.bl] = 1
        return window_mask.to(device)

    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder

        query = self.query_conv(x1)
        key = self.key_conv(x1)

        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    def _normal_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _block_wise_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()

        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                                  torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)], dim=-1)

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb,
                                                                                                     1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)

        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                                  torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)], dim=-1)

        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)

        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k,
                       torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v,
                       torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask,
                                  torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)

        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)],
                      dim=0)  # special case when self.bl = 1
        v = torch.cat([v[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)], dim=0)
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat(
            [padding_mask[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)],
            dim=0)  # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask

        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]

        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head, future_window):
        super(MultiHeadAttLayer, self).__init__()
        #         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, future_window=future_window))
             for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out


class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.layer(x)


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, future_window):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type,
                                  stage=stage, future_window=future_window)  # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        norm = self.instance_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.alpha * self.att_layer(norm, f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)

    #         self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]


class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha,
                 future_window):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha, future_window=future_window)
             for i in  # 2**i
             range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha, future_window):
        super(Decoder, self).__init__()  # self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, future_window=future_window)
             for i in  # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate,
                 future_window):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate,
                               att_type='sliding_att', alpha=1, future_window=future_window)
        self.decoders = nn.ModuleList([copy.deepcopy(
            Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att',
                    alpha=exponential_descrease(s), future_window=future_window)) for s in
            range(num_decoders)])  # num_decoders

    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)

        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate,
                 future_window, decoders_num=3):
        self.model = MyTransformer(decoders_num, num_layers, r1, r2, num_f_maps, input_dim, num_classes,
                                   channel_masking_rate, future_window=future_window)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, batch_gen_tst, train_data_len, run):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print('LR:{}'.format(learning_rate))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            es = 0
            overlap = [.1, .25, .5]
            tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
            pbar = tqdm(total=train_data_len)
            i = 0

            while batch_gen.has_next():
                i += 1
                batch_input, batch_target, mask, vids = batch_gen.next_batch(batch_size, False)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                # length = 10
                ps = self.model(batch_input, mask)
                # ps = self.model(batch_input[:, :, :length], mask[:, :, :length])
                # batch_target = batch_target[:, :length]
                # mask = mask[:, :, :length]

                loss = 0
                for p in ps:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                pbar.update(1)

                _, predicted = torch.max(ps.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                predicted = predicted.detach().cpu().numpy().flatten()
                batch_target = batch_target.detach().cpu().numpy().flatten()
                es += edit_score(predicted, batch_target)
                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(predicted, batch_target, overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
                total += torch.sum(mask[:, 0, :]).item()
                torch.cuda.empty_cache()

            pbar.close()
            scheduler.step(epoch_loss)
            batch_gen.reset()
            es = float(es) / i
            f1s = self.f1s(overlap, tp, fp, fn)
            train_loss = epoch_loss / len(batch_gen.list_of_examples)
            acc = (float(correct) / total) * 100
            print("[epoch %d]: epoch loss = %f acc = %.4f edit_score = %.4f F1@10 = %.4f F1@25 = %.4f F1@50 = %.4f" %
                  (epoch + 1, train_loss, acc, es, f1s[0], f1s[1], f1s[2]))

            loss_tst, acc_tst, es_tst, f1s_tst = self.test(batch_gen_tst, epoch, run)

            run.log({"train/loss": train_loss, "train/accuracy": acc, "train/edit_score": es,
                     "train/F1@10": f1s[0], "train/F1@25": f1s[1], "train/F1@50": f1s[2],

                     "validation/loss": loss_tst, "validation/accuracy": acc_tst, "validation/edit_score": es_tst,
                     "validation/F1@10": f1s_tst[0], "validation/F1@25": f1s_tst[1], "validation/F1@50": f1s_tst[2]
                     })

            torch.save(self.model.state_dict(),
                       save_dir + "/epoch-" + str(epoch + 1) + f"-val_loss-{loss_tst}_val_acc-{acc_tst}.model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

    def test(self, batch_gen_tst, epoch, run):
        self.model.eval()
        correct = 0
        total = 0
        es_tst = 0
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
        if_warp = False  # When testing, always false
        i = 0
        loss = 0
        with torch.no_grad():
            while batch_gen_tst.has_next():
                i += 1
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1, if_warp)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                p = self.model(batch_input, mask)
                for pp in p:
                    loss += self.ce(pp.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(pp[:, :, 1:], dim=1), F.log_softmax(pp.detach()[:, :, :-1], dim=1)),
                        min=0,
                        max=16) * mask[:, :, 1:])
                _, predicted = torch.max(p.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                predicted = predicted.detach().cpu().numpy().flatten()
                batch_target = batch_target.detach().cpu().numpy().flatten()
                es_tst += edit_score(predicted, batch_target)
                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(predicted, batch_target, overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
                total += torch.sum(mask[:, 0, :]).item()

        acc_tst = (float(correct) / total) * 100
        es_tst = float(es_tst) / i
        f1s_tst = self.f1s(overlap, tp, fp, fn)
        loss_tst = loss / len(batch_gen_tst.list_of_examples)
        print("[epoch %d]: tst loss = %.4f acc = %.4f edit_score = %.4f F1@10 = %.4f F1@25 = %.4f F1@50 = %.4f" %
              (epoch + 1, loss_tst, acc_tst, es_tst, f1s_tst[0], f1s_tst[1], f1s_tst[2]))
        # run.log({"validation/loss": loss_tst, "validation/accuracy": acc_tst, "validation/edit_score": es_tst,
        #          "validation/F1@10": f1s_tst[0], "validation/F1@25": f1s_tst[1], "validation/F1@50": f1s_tst[2]})

        self.model.train()
        batch_gen_tst.reset()
        return loss_tst, acc_tst, es_tst, f1s_tst

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate, run,
                future_window=0, plot=None, future_description=None):
        self.model.eval()
        correct = 0
        total = 0
        es = 0
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
        if_warp = False  # When testing, always false
        i = 0
        with torch.no_grad():
            self.model.to(device)
            # self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            self.model.load_state_dict(torch.load(model_dir + "/best.model"))
            batch_gen_tst.reset()
            my_data = []
            vid_names = []
            all_images = []

            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                vid = vids[0]
                vid_names.append(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                images = []

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()

                    batch_target = batch_target.squeeze()
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()

                    img = segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
                                                       confidence.tolist(),
                                                       batch_target.tolist(), predicted.tolist())
                    images.append(img)
                all_images.append(images)

                _, predicted = torch.max(predictions.data[-1], 1)
                predicted = predicted.detach().cpu().flatten()
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                predicted = predicted.numpy()
                batch_target = batch_target.detach().cpu().numpy().flatten()
                es += edit_score(predicted, batch_target)
                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(predicted, batch_target, overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
                total += torch.sum(mask[:, 0, :]).item()

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
            es = float(es) / i
            f1s = self.f1s(overlap, tp, fp, fn)
            acc = (float(correct) / total) * 100
            print(f"future_window{future_window}", " accuracy: ", acc, " edit_score: ", es,
                  " F1@10: ", f1s[0], " F1@25: ", f1s[1], " F1@50: ", f1s[2])
            future_description = 3 if str(future_description) == 'all future' else future_window / 15
            plot['accuracy'].append([future_description, acc])
            plot['edit distance'].append([future_description, es])
            plot['F1@10'].append([future_description, f1s[0]])
            plot['F1@25'].append([future_description, f1s[1]])
            plot['F1@50'].append([future_description, f1s[2]])
            argmin = min(range(len(vid_names)), key=lambda i: vid_names[i])
            images = all_images[argmin]
            return [vid_names[argmin], future_description, wandb.Image(images[0]), wandb.Image(images[1]),
                    wandb.Image(images[2])], plot

    @staticmethod
    def f1s(overlap, tp, fp, fn):
        f1s = np.array([0, 0, 0], dtype=float)
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s] + fp[s])
            recall = tp[s] / float(tp[s] + fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            f1s[s] = f1
        return f1s


if __name__ == '__main__':
    pass
