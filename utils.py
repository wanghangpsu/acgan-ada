# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def get_roc(clean , attack, thresholds):
    detection_rate = []
    false_alarm = []
    for threshold in thresholds:
        dr = 0
        fl = 0
        for i in range(len(clean)):
            if clean[i]>threshold:
                fl += 1
            if attack[i] > threshold:
                dr += 1
        detection_rate.append(dr/len(attack))
        false_alarm.append(fl/len(clean))
    return detection_rate, false_alarm


