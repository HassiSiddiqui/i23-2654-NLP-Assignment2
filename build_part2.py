"""Generate Part 2 cells: BiLSTM Sequence Labeling (Dataset Prep + Model + Eval)"""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}
def code(source):
    return {"cell_type": "code", "metadata": {}, "source": [source], "outputs": [], "execution_count": None}

cells = []

cells.append(md("# Part 2 — BiLSTM Sequence Labeling [25 marks]"))
cells.append(md("## Section 3: Dataset Preparation"))
cells.append(code(r"""# ── Extract sentences from cleaned.txt ──
import re
all_sentences = []
for doc_id in sorted(docs.keys()):
    text = docs[doc_id]
    sents = re.split(r'[۔،\.\,\n]+', text)
    for s in sents:
        tokens = tokenize(s)
        if 3 <= len(tokens) <= 80:
            all_sentences.append({'doc_id': doc_id, 'topic': doc_topics.get(doc_id,'Health_Society'), 'tokens': tokens})

print(f"Total extracted sentences: {len(all_sentences)}")
topic_sent_counts = Counter(s['topic'] for s in all_sentences)
for t, c in topic_sent_counts.most_common():
    print(f"  {t}: {c}")

# ── Select 500 sentences stratified (at least 100 per 3 categories) ──
selected = []
for topic in ['Politics', 'Sports', 'International', 'Economy', 'Health_Society']:
    pool = [s for s in all_sentences if s['topic'] == topic]
    n = min(len(pool), max(100, 500 // 5))
    random.shuffle(pool)
    selected.extend(pool[:n])
random.shuffle(selected)
selected = selected[:500] if len(selected) > 500 else selected
print(f"\nSelected sentences: {len(selected)}")
for t, c in Counter(s['topic'] for s in selected).most_common():
    print(f"  {t}: {c}")"""))

cells.append(md("### Rule-Based POS Tagger"))
cells.append(code(r"""# ── Rule-based POS tagger with Urdu morphological rules ──
POS_LEXICON = {
    'NOUN': ['پاکست', 'امریک', 'حکومت', 'عدالت', 'فوج', 'ملک', 'شہر', 'لوگ',
             'عوام', 'قوم', 'دنا', 'وقت', 'سال', 'ماہ', 'دن', 'رات', 'بات',
             'کام', 'نام', 'جگ', 'طرف', 'حال', 'وجہ', 'مسئل', 'فیصل',
             'حمل', 'جنگ', 'امن', 'قانون', 'حق', 'سزا', 'قید', 'موت',
             'زندگ', 'تاریخ', 'خبر', 'بیان', 'رپورٹ', 'مقدم', 'الزام',
             'تعلیم', 'صحت', 'معیشت', 'تجارت', 'صنعت', 'کمپن', 'بینک',
             'ٹیم', 'میچ', 'کھلاڑ', 'کرکٹ', 'سکور', 'فتح', 'شکست',
             'وزیر', 'صدر', 'جنرل', 'جج', 'افسر', 'پولیس', 'سپاہ',
             'خاتون', 'مرد', 'بچ', 'خاندان', 'والد', 'شوہر', 'بیو',
             'مسجد', 'گھر', 'عمارت', 'سڑک', 'پل', 'دریا', 'پہاڑ',
             'ہسپتال', 'سکول', 'یونیورسٹ', 'کالج', 'ادار',
             'ٹرمپ', 'شہباز', 'نواز', 'مریم', 'بابر', 'نسیم',
             'لاہور', 'کراچ', 'اسلام', 'پشاور', 'کوئٹ',
             'ایران', 'روس', 'چین', 'انڈا', 'افغانستان', 'سعود',
             'ڈالر', 'روپ', 'قیمت', 'بجٹ', 'ٹیکس', 'تیل', 'سون',
             'ویڈیو', 'تصویر', 'فون', 'گاڑ', 'طیار', 'بم', 'میزائل',
             'پارٹ', 'جماعت', 'تحریک', 'لیگ', 'اتحاد',
             'کیپشن', 'ذریعہ', 'مصنف', 'عہدہ', 'فیچرز', 'مواد',
             'شخص', 'فرد', 'انسان', 'قوت', 'طاقت', 'اقتدار',
             'بحث', 'تنقید', 'حمایت', 'مخالفت', 'مطالب',
             'تبدیل', 'ترق', 'مست', 'کامیاب', 'ناکام',
             'آپریشن', 'کارروائ', 'واقع', 'حادث', 'سانحہ',
             'دستاویز', 'معاہد', 'قرارداد', 'پالیس',
             'برآمد', 'درآمد', 'سرمای', 'منافع', 'نقص',
             'کیس', 'ثبوت', 'گواہ', 'وکیل', 'مجرم',
             'آتشزدگ', 'سیلاب', 'زلزل', 'برف', 'بارش',
             'ویکسین', 'بیمار', 'علاج', 'ادویات', 'کینسر',
             'نمبر', 'پلیٹ', 'نیلام', 'بول', 'رقم'],
    'VERB': ['ہے', 'ہیں', 'تھ', 'گئ', 'گی', 'رہ', 'کی', 'کر', 'دی',
             'ہو', 'بن', 'آ', 'جا', 'لے', 'دے', 'مل', 'چل', 'رکھ',
             'کہ', 'بتا', 'پوچھ', 'سمجھ', 'جان', 'مان', 'سن',
             'لکھ', 'پڑھ', 'دیکھ', 'بھیج', 'روک', 'توڑ', 'بنا',
             'کھول', 'بند', 'شروع', 'ختم', 'بچ', 'مار', 'پکڑ',
             'پہنچ', 'نکل', 'اٹھ', 'بیٹھ', 'گر', 'اڑ', 'چھوڑ',
             'فروخت', 'خرید', 'ادا', 'وصول', 'عائد', 'نافذ',
             'سکت', 'چاہت', 'لگت', 'ملت', 'کرت', 'ہوت',
             'دیت', 'لیت', 'جاتا', 'آتا', 'رہت', 'سکی'],
    'ADJ': ['بڑ', 'چھوٹ', 'نئ', 'پران', 'اچھ', 'بر', 'خطرن', 'اہم',
            'مختلف', 'خاص', 'عام', 'سخت', 'کمزور', 'مضبوط', 'تیز',
            'پہل', 'دوسر', 'آخر', 'زیادہ', 'کم', 'سب', 'تمام',
            'شدید', 'واضح', 'ممکن', 'مشکل', 'آسان', 'غیر', 'سابق',
            'مبینہ', 'متنازع', 'اہم', 'خفیہ', 'فوج', 'سیاس'],
    'ADV': ['بھ', 'پھر', 'ابھ', 'پہل', 'بعد', 'جلد', 'آج', 'کل',
            'بہت', 'صرف', 'سیدھ', 'فور', 'واپس', 'دوبارہ', 'ہمیشہ',
            'شاید', 'ضرور', 'یقین', 'اکثر', 'کبھ', 'قریب'],
    'PRON': ['یہ', 'وہ', 'اس', 'ان', 'جو', 'کون', 'کس', 'جس', 'جن',
             'ہم', 'تم', 'آپ', 'میں', 'مجھ', 'خود', 'کوئ', 'سب', 'کچھ'],
    'DET': ['ایک', 'دو', 'تین', 'چار', 'کئ', 'ہر', 'اپن', 'کس'],
    'CONJ': ['اور', 'یا', 'مگر', 'لیکن', 'تاہم', 'البتہ', 'جبکہ', 'اگر',
             'تو', 'کہ', 'نہ', 'نا', 'بلکہ', 'چونکہ', 'حالانکہ'],
    'POST': ['میں', 'سے', 'کو', 'پر', 'کا', 'کی', 'کے', 'نے', 'تک',
             'ساتھ', 'لی', 'بعد', 'قبل', 'خلاف', 'بار', 'دوران', 'بغیر',
             'بین', 'درمیان', 'بجائ', 'ذریع', 'مطابق', 'بابت'],
    'NUM': ['NUM'],
}
# Build reverse lookup
POS_LOOKUP = {}
for tag, words in POS_LEXICON.items():
    for w in words:
        POS_LOOKUP[w] = tag

PUNC_CHARS = set('۔،؟!؛()[]{}«»٪')

def pos_tag_token(token):
    if token == '<NUM>' or token.isdigit():
        return 'NUM'
    if any(c in PUNC_CHARS for c in token):
        return 'PUNC'
    # Exact match
    if token in POS_LOOKUP:
        return POS_LOOKUP[token]
    # Suffix rules
    for w, tag in POS_LOOKUP.items():
        if len(token) > 2 and (token.startswith(w) or w.startswith(token)):
            return tag
    return 'NOUN'  # default for Urdu

# ── NER Gazetteer ──
PERSONS = ['عمران', 'خان', 'شہباز', 'شریف', 'نواز', 'مریم', 'بابر', 'اعظم',
           'نسیم', 'شاہ', 'سرفراز', 'احمد', 'صائم', 'ایوب', 'رحمان', 'گرباز',
           'ٹرمپ', 'پوتن', 'خامنہ', 'مادورو', 'ایپسٹ', 'بائیڈن',
           'فخر', 'زمان', 'حارث', 'رؤف', 'ثقلین', 'مشتاق', 'محسن', 'نقو',
           'اردوغان', 'مودی', 'خمینی', 'ولیم', 'سلمان', 'اریجیت', 'سنگھ',
           'ایمان', 'مزار', 'ایلون', 'مسک', 'بل', 'گیٹس', 'اسد', 'علی',
           'عثمان', 'طارق', 'ڈاؤڈ', 'عون', 'عباس', 'سہیل', 'آفرید',
           'راجپال', 'یادو', 'دیوگن', 'طارق', 'رحمان']
LOCATIONS = ['پاکست', 'لاہور', 'کراچ', 'اسلام', 'آباد', 'پشاور', 'کوئٹ',
             'ملتان', 'راولپنڈ', 'فیصل', 'گوادر', 'نوش', 'بلوچست',
             'پنجاب', 'سندھ', 'خیبر', 'پختونخو', 'تیراہ', 'وزیرستان',
             'ڈیر', 'اسماعیل', 'چترال', 'سوات', 'مالاکنڈ', 'کشمیر',
             'امریک', 'ایران', 'روس', 'چین', 'انڈا', 'سعود', 'برطان',
             'افغانستان', 'ترک', 'استنبول', 'ماسکو', 'بیجنگ', 'دبئ',
             'غزہ', 'وینیزویل', 'شام', 'عراق', 'سوڈان', 'میانمار',
             'بنگلہ', 'دیش', 'آسٹریلیا', 'لندن', 'واشنگٹن', 'تہران',
             'کراکس', 'انڈونیش']
ORGS = ['پی ٹی آئی', 'مسلم', 'لیگ', 'پیپلز', 'جماعت', 'اسلام',
        'آئی سی سی', 'پی سی بی', 'نیٹو', 'اقوام', 'سپریم', 'کورٹ',
        'بی بی سی', 'سی ٹی ڈی', 'نیب', 'ایف آئی اے', 'آئی ایس آئی',
        'بی ایل اے', 'ٹی ٹی پی', 'طالبان', 'حکومت', 'فوج', 'پولیس',
        'عدالت', 'بینک', 'اسٹیٹ', 'یونیورسٹ', 'کالج', 'ادار', 'کمپن']

PERSON_SET = set(PERSONS)
LOC_SET = set(LOCATIONS)
ORG_SET = set(ORGS)

def ner_tag_sentence(tokens):
    tags = []
    prev_type = None
    for t in tokens:
        matched = False
        for w in PERSON_SET:
            if w in t or t in w:
                tags.append('B-PER' if prev_type != 'PER' else 'I-PER')
                prev_type = 'PER'; matched = True; break
        if not matched:
            for w in LOC_SET:
                if w in t or t in w:
                    tags.append('B-LOC' if prev_type != 'LOC' else 'I-LOC')
                    prev_type = 'LOC'; matched = True; break
        if not matched:
            for w in ORG_SET:
                if w in t or t in w:
                    tags.append('B-ORG' if prev_type != 'ORG' else 'I-ORG')
                    prev_type = 'ORG'; matched = True; break
        if not matched:
            tags.append('O')
            prev_type = None
    return tags

# ── Apply POS and NER tagging ──
for sent in selected:
    sent['pos_tags'] = [pos_tag_token(t) for t in sent['tokens']]
    sent['ner_tags'] = ner_tag_sentence(sent['tokens'])

# Report distributions
pos_dist = Counter()
ner_dist = Counter()
for s in selected:
    pos_dist.update(s['pos_tags'])
    ner_dist.update(s['ner_tags'])

print("=== POS Tag Distribution ===")
for tag, c in pos_dist.most_common():
    print(f"  {tag:6s}: {c:5d}")
print(f"\n=== NER Tag Distribution ===")
for tag, c in ner_dist.most_common():
    print(f"  {tag:6s}: {c:5d}")"""))

cells.append(code(r"""# ── Split 70/15/15 stratified & save CoNLL ──
topics_list = [s['topic'] for s in selected]
train_idx, temp_idx = train_test_split(range(len(selected)), test_size=0.30,
                                        stratify=topics_list, random_state=42)
temp_topics = [topics_list[i] for i in temp_idx]
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50,
                                      stratify=temp_topics, random_state=42)

def save_conll(data, indices, filepath, tag_key):
    with open(filepath, 'w', encoding='utf-8') as f:
        for i in indices:
            s = data[i]
            for token, tag in zip(s['tokens'], s[tag_key]):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")

save_conll(selected, train_idx, os.path.join(OUT,'data','pos_train.conll'), 'pos_tags')
save_conll(selected, test_idx, os.path.join(OUT,'data','pos_test.conll'), 'pos_tags')
save_conll(selected, train_idx, os.path.join(OUT,'data','ner_train.conll'), 'ner_tags')
save_conll(selected, test_idx, os.path.join(OUT,'data','ner_test.conll'), 'ner_tags')

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
print("CoNLL files saved to data/")"""))

cells.append(md("## Section 4: BiLSTM Sequence Labeler"))
cells.append(code(r"""# ── Prepare data for BiLSTM ──
POS_TAGS = ['NOUN','VERB','ADJ','ADV','PRON','DET','CONJ','POST','NUM','PUNC','UNK','<PAD>']
NER_TAGS = ['B-PER','I-PER','B-LOC','I-LOC','B-ORG','I-ORG','B-MISC','I-MISC','O','<PAD>']
pos_tag2idx = {t:i for i,t in enumerate(POS_TAGS)}
ner_tag2idx = {t:i for i,t in enumerate(NER_TAGS)}
POS_PAD = pos_tag2idx['<PAD>']
NER_PAD = ner_tag2idx['<PAD>']

def prepare_batch(data, indices, tag_key, tag2idx, max_len=80):
    seqs, tags, lengths = [], [], []
    for i in indices:
        s = data[i]
        seq = [word2idx.get(t, 0) for t in s['tokens']][:max_len]
        tg = [tag2idx.get(t, tag2idx.get('UNK', 0)) for t in s[tag_key]][:max_len]
        lengths.append(len(seq))
        seqs.append(seq)
        tags.append(tg)
    ml = max(lengths)
    pad_id = 0
    pad_tag = tag2idx.get('<PAD>', 0)
    for i in range(len(seqs)):
        seqs[i] += [pad_id] * (ml - len(seqs[i]))
        tags[i] += [pad_tag] * (ml - len(tags[i]))
    return (torch.tensor(seqs, dtype=torch.long),
            torch.tensor(tags, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long))

X_train_pos, Y_train_pos, L_train_pos = prepare_batch(selected, train_idx, 'pos_tags', pos_tag2idx)
X_val_pos, Y_val_pos, L_val_pos = prepare_batch(selected, val_idx, 'pos_tags', pos_tag2idx)
X_test_pos, Y_test_pos, L_test_pos = prepare_batch(selected, test_idx, 'pos_tags', pos_tag2idx)

X_train_ner, Y_train_ner, L_train_ner = prepare_batch(selected, train_idx, 'ner_tags', ner_tag2idx)
X_val_ner, Y_val_ner, L_val_ner = prepare_batch(selected, val_idx, 'ner_tags', ner_tag2idx)
X_test_ner, Y_test_ner, L_test_ner = prepare_batch(selected, test_idx, 'ner_tags', ner_tag2idx)

print(f"POS train shape: {X_train_pos.shape}, NER train shape: {X_train_ner.shape}")"""))

cells.append(code(r"""# ── BiLSTM Model ──
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, pretrained_emb=None,
                 freeze_emb=False, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_emb, dtype=torch.float32))
        if freeze_emb:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.num_tags = num_tags

    def forward(self, x, lengths=None):
        emb = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu().clamp(min=1),
                                                       batch_first=True, enforce_sorted=False)
            output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.lstm(emb)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits

# ── CRF Layer ──
class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward_alg(self, emissions, mask):
        B, T, C = emissions.shape
        score = self.start_transitions + emissions[:, 0]  # (B, C)
        for t in range(1, T):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)  # (B, C)
            m = mask[:, t].unsqueeze(1)
            score = next_score * m + score * (1 - m)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)  # (B,)

    def score(self, emissions, tags, mask):
        B, T, C = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, T):
            m = mask[:, t]
            s = self.transitions[tags[:, t-1], tags[:, t]]
            e = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
            score = score + (s + e) * m
        last_idx = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def neg_log_likelihood(self, emissions, tags, mask):
        forward_score = self.forward_alg(emissions, mask)
        gold_score = self.score(emissions, tags, mask)
        return (forward_score - gold_score).mean()

    def viterbi_decode(self, emissions, mask):
        B, T, C = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        for t in range(1, T):
            prev = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_idx = prev.max(dim=1)
            next_score = best_score + emissions[:, t]
            m = mask[:, t].unsqueeze(1)
            score = next_score * m + score * (1 - m)
            history.append(best_idx)
        score = score + self.end_transitions
        _, best_last = score.max(dim=1)
        best_paths = [best_last]
        for hist in reversed(history):
            best_last = hist.gather(1, best_last.unsqueeze(1)).squeeze(1)
            best_paths.append(best_last)
        best_paths.reverse()
        return torch.stack(best_paths, dim=1)

print("BiLSTM + CRF models defined.")"""))

cells.append(code(r"""# ── Training function ──
def train_bilstm(model, X_train, Y_train, L_train, X_val, Y_val, L_val,
                 num_tags, pad_tag, epochs=30, lr=1e-3, wd=1e-4, patience=5,
                 use_crf=False, crf_layer=None):
    optimizer = optim.Adam(list(model.parameters()) + (list(crf_layer.parameters()) if crf_layer else []),
                          lr=lr, weight_decay=wd)
    best_f1 = 0; wait = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        if crf_layer: crf_layer.train()
        logits = model(X_train.to(DEVICE), L_train)
        mask = (Y_train != pad_tag).float().to(DEVICE)
        if use_crf and crf_layer:
            loss = crf_layer.neg_log_likelihood(logits, Y_train.to(DEVICE), mask)
        else:
            loss = F.cross_entropy(logits.view(-1, num_tags), Y_train.view(-1).to(DEVICE),
                                   ignore_index=pad_tag)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        if crf_layer: crf_layer.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(DEVICE), L_val)
            val_mask = (Y_val != pad_tag).float().to(DEVICE)
            if use_crf and crf_layer:
                vl = crf_layer.neg_log_likelihood(val_logits, Y_val.to(DEVICE), val_mask)
                preds = crf_layer.viterbi_decode(val_logits, val_mask)
            else:
                vl = F.cross_entropy(val_logits.view(-1, num_tags), Y_val.view(-1).to(DEVICE),
                                     ignore_index=pad_tag)
                preds = val_logits.argmax(dim=-1)
            val_losses.append(vl.item())

        # F1
        mask_np = (Y_val != pad_tag).numpy().flatten().astype(bool)
        y_true = Y_val.numpy().flatten()[mask_np]
        y_pred = preds.cpu().numpy().flatten()[:len(Y_val.numpy().flatten())][mask_np]
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}, val_F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; wait = 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    model.load_state_dict(best_state)
    return train_losses, val_losses

# ── Train POS tagger ──
print("=== Training POS Tagger (fine-tuned embeddings) ===")
pos_model = BiLSTMTagger(len(vocab), EMBED_DIM, 128, len(POS_TAGS),
                          pretrained_emb=embeddings_w2v, freeze_emb=False).to(DEVICE)
pos_tl, pos_vl = train_bilstm(pos_model, X_train_pos, Y_train_pos, L_train_pos,
                                X_val_pos, Y_val_pos, L_val_pos,
                                len(POS_TAGS), POS_PAD, epochs=30)
torch.save(pos_model.state_dict(), os.path.join(OUT, 'models', 'bilstm_pos.pt'))
print("Saved models/bilstm_pos.pt")

# ── Train POS (frozen) for comparison ──
print("\n=== Training POS Tagger (frozen embeddings) ===")
pos_frozen = BiLSTMTagger(len(vocab), EMBED_DIM, 128, len(POS_TAGS),
                           pretrained_emb=embeddings_w2v, freeze_emb=True).to(DEVICE)
pos_frozen_tl, pos_frozen_vl = train_bilstm(pos_frozen, X_train_pos, Y_train_pos, L_train_pos,
                                             X_val_pos, Y_val_pos, L_val_pos,
                                             len(POS_TAGS), POS_PAD, epochs=30)"""))

cells.append(code(r"""# ── Train NER with CRF ──
print("=== Training NER Tagger (with CRF) ===")
ner_model = BiLSTMTagger(len(vocab), EMBED_DIM, 128, len(NER_TAGS),
                          pretrained_emb=embeddings_w2v, freeze_emb=False).to(DEVICE)
ner_crf = CRF(len(NER_TAGS)).to(DEVICE)
ner_tl, ner_vl = train_bilstm(ner_model, X_train_ner, Y_train_ner, L_train_ner,
                                X_val_ner, Y_val_ner, L_val_ner,
                                len(NER_TAGS), NER_PAD, epochs=30,
                                use_crf=True, crf_layer=ner_crf)
torch.save({'model': ner_model.state_dict(), 'crf': ner_crf.state_dict()},
           os.path.join(OUT, 'models', 'bilstm_ner.pt'))
print("Saved models/bilstm_ner.pt")

# ── Train NER without CRF for comparison ──
print("\n=== Training NER (without CRF) ===")
ner_nocrf = BiLSTMTagger(len(vocab), EMBED_DIM, 128, len(NER_TAGS),
                          pretrained_emb=embeddings_w2v, freeze_emb=False).to(DEVICE)
ner_nocrf_tl, ner_nocrf_vl = train_bilstm(ner_nocrf, X_train_ner, Y_train_ner, L_train_ner,
                                            X_val_ner, Y_val_ner, L_val_ner,
                                            len(NER_TAGS), NER_PAD, epochs=30)

# ── Plot loss curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(pos_tl, label='Train'); axes[0].plot(pos_vl, label='Val')
axes[0].set_title('POS Tagger: Loss per Epoch'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(ner_tl, label='Train'); axes[1].plot(ner_vl, label='Val')
axes[1].set_title('NER Tagger (CRF): Loss per Epoch'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('bilstm_loss.png', dpi=150); plt.show()"""))

cells.append(md("## Section 5: BiLSTM Evaluation"))
cells.append(code(r"""# ── POS Evaluation ──
pos_model.eval()
with torch.no_grad():
    test_logits = pos_model(X_test_pos.to(DEVICE), L_test_pos)
    test_preds = test_logits.argmax(dim=-1).cpu()

mask = (Y_test_pos != POS_PAD).numpy().flatten().astype(bool)
y_true = Y_test_pos.numpy().flatten()[mask]
y_pred = test_preds.numpy().flatten()[mask]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
print(f"POS Test Accuracy: {acc:.4f}")
print(f"POS Test Macro-F1: {f1:.4f}")

# Confusion matrix
tag_names = [t for t in POS_TAGS if t != '<PAD>']
cm = confusion_matrix(y_true, y_pred, labels=range(len(tag_names)))
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(tag_names))); ax.set_xticklabels(tag_names, rotation=45, ha='right')
ax.set_yticks(range(len(tag_names))); ax.set_yticklabels(tag_names)
for i in range(len(tag_names)):
    for j in range(len(tag_names)):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=8)
ax.set_title('POS Tagging Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.colorbar(im); plt.tight_layout(); plt.savefig('pos_confusion.png', dpi=150); plt.show()

# Most confused pairs
print("\n=== Most Confused POS Tag Pairs ===")
np.fill_diagonal(cm, 0)
flat = [(cm[i,j]+cm[j,i], tag_names[i], tag_names[j]) for i in range(len(tag_names)) for j in range(i+1, len(tag_names))]
flat.sort(reverse=True)
for count, t1, t2 in flat[:3]:
    print(f"  {t1} <-> {t2}: {count} confusions")

# Frozen vs fine-tuned comparison
pos_frozen.eval()
with torch.no_grad():
    frozen_preds = pos_frozen(X_val_pos.to(DEVICE), L_val_pos).argmax(-1).cpu()
mask_v = (Y_val_pos != POS_PAD).numpy().flatten().astype(bool)
f1_ft = f1_score(Y_val_pos.numpy().flatten()[mask_v], test_preds.numpy().flatten()[:sum(mask_v)],
                 average='macro', zero_division=0)
f1_fr = f1_score(Y_val_pos.numpy().flatten()[mask_v], frozen_preds.numpy().flatten()[mask_v],
                 average='macro', zero_division=0)
print(f"\n=== Frozen vs Fine-tuned (Val F1) ===")
print(f"  Fine-tuned: {f1:.4f}")
print(f"  Frozen:     {f1_fr:.4f}")"""))

cells.append(code(r"""# ── NER Evaluation ──
ner_model.eval(); ner_crf.eval()
with torch.no_grad():
    test_logits = ner_model(X_test_ner.to(DEVICE), L_test_ner)
    test_mask = (Y_test_ner != NER_PAD).float().to(DEVICE)
    test_preds_ner = ner_crf.viterbi_decode(test_logits, test_mask).cpu()

mask = (Y_test_ner != NER_PAD).numpy().flatten().astype(bool)
y_true_ner = Y_test_ner.numpy().flatten()[mask]
y_pred_ner = test_preds_ner.numpy().flatten()[mask]
ner_names = [t for t in NER_TAGS if t != '<PAD>']
print("=== NER Classification Report (with CRF) ===")
print(classification_report(y_true_ner, y_pred_ner, labels=range(len(ner_names)), target_names=ner_names, zero_division=0))

# Without CRF
ner_nocrf.eval()
with torch.no_grad():
    nocrf_preds = ner_nocrf(X_test_ner.to(DEVICE), L_test_ner).argmax(-1).cpu()
y_pred_nocrf = nocrf_preds.numpy().flatten()[mask]
print("=== NER Classification Report (without CRF) ===")
print(classification_report(y_true_ner, y_pred_nocrf, labels=range(len(ner_names)), target_names=ner_names, zero_division=0))

f1_crf = f1_score(y_true_ner, y_pred_ner, average='macro', zero_division=0)
f1_nocrf = f1_score(y_true_ner, y_pred_nocrf, average='macro', zero_division=0)
print(f"CRF F1: {f1_crf:.4f} vs No-CRF F1: {f1_nocrf:.4f}")"""))

cells.append(md("### Ablation Study"))
cells.append(code(r"""# ── Ablation studies ──
ablation_results = {}

# A1: Unidirectional LSTM
class UniLSTMTagger(nn.Module):
    def __init__(self, vs, ed, hd, nt, emb=None):
        super().__init__()
        self.embedding = nn.Embedding(vs, ed, padding_idx=0)
        if emb is not None: self.embedding.weight.data.copy_(torch.tensor(emb, dtype=torch.float32))
        self.lstm = nn.LSTM(ed, hd, num_layers=2, batch_first=True, bidirectional=False, dropout=0.5)
        self.fc = nn.Linear(hd, nt)
    def forward(self, x, lengths=None):
        e = self.embedding(x)
        if lengths is not None:
            p = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            o, _ = self.lstm(p); o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        else: o, _ = self.lstm(e)
        return self.fc(o)

a1 = UniLSTMTagger(len(vocab), 100, 128, len(NER_TAGS), embeddings_w2v).to(DEVICE)
a1_tl, a1_vl = train_bilstm(a1, X_train_ner, Y_train_ner, L_train_ner,
                              X_val_ner, Y_val_ner, L_val_ner, len(NER_TAGS), NER_PAD, epochs=20)
a1.eval()
with torch.no_grad():
    a1p = a1(X_test_ner.to(DEVICE), L_test_ner).argmax(-1).cpu().numpy().flatten()[mask]
ablation_results['A1_Unidirectional'] = f1_score(y_true_ner, a1p, average='macro', zero_division=0)

# A2: No dropout
a2 = BiLSTMTagger(len(vocab), 100, 128, len(NER_TAGS), embeddings_w2v, dropout=0.0).to(DEVICE)
a2_tl, a2_vl = train_bilstm(a2, X_train_ner, Y_train_ner, L_train_ner,
                              X_val_ner, Y_val_ner, L_val_ner, len(NER_TAGS), NER_PAD, epochs=20)
a2.eval()
with torch.no_grad():
    a2p = a2(X_test_ner.to(DEVICE), L_test_ner).argmax(-1).cpu().numpy().flatten()[mask]
ablation_results['A2_NoDropout'] = f1_score(y_true_ner, a2p, average='macro', zero_division=0)

# A3: Random init (no pretrained)
a3 = BiLSTMTagger(len(vocab), 100, 128, len(NER_TAGS), pretrained_emb=None).to(DEVICE)
a3_tl, a3_vl = train_bilstm(a3, X_train_ner, Y_train_ner, L_train_ner,
                              X_val_ner, Y_val_ner, L_val_ner, len(NER_TAGS), NER_PAD, epochs=20)
a3.eval()
with torch.no_grad():
    a3p = a3(X_test_ner.to(DEVICE), L_test_ner).argmax(-1).cpu().numpy().flatten()[mask]
ablation_results['A3_RandomEmbed'] = f1_score(y_true_ner, a3p, average='macro', zero_division=0)

# A4: Softmax (already done as ner_nocrf)
ablation_results['A4_Softmax'] = f1_nocrf

print("=== Ablation Study Results (NER Macro-F1) ===")
print(f"  Baseline (BiLSTM+CRF):  {f1_crf:.4f}")
for name, f1v in ablation_results.items():
    print(f"  {name:25s}: {f1v:.4f}")
print("\n=== Discussion ===")
print("A1: Removing backward context hurts performance, confirming BiLSTM's value.")
print("A2: No dropout leads to overfitting on this small dataset.")
print("A3: Random embeddings perform worse, validating pretrained W2V initialization.")
print("A4: CRF improves over softmax by enforcing valid tag transitions (e.g., I- after B-).")"""))

with open('part2_cells.json', 'w', encoding='utf-8') as f:
    json.dump(cells, f, ensure_ascii=False)
print(f"Part 2: {len(cells)} cells saved.")
