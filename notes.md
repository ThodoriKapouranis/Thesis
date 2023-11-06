Wet soil has increased electrical conductivity, causing the S1-SAR imagery to become brighter. Once water starts to accumulate, the specular reflection of microwaves causes the signal to drop.

With forested canopies, the water bounces off the water and double bounces on tree trunks or other vertical terrain and bounces back towards the sensor. This causes a spike in signal.



| Scenario | Description | Number of channels |
|   --- |               ---     |   --- |
|   I   |   Co-event intensity  |   2   |
|   II  |   Pre + Co intensity  |   4   |
|   III |   Pre + Co Int + Coh  |   6   |


**Sri-Lanka chips are not used for training. Save for model generalization analysis**
44,178,776
**Highly unbalanced classes but do not assign class weights during training? Why??**

# Val = Sri-Lanka Holdout
# Hand = Hand labelled Testing
train_ds, test_ds, val_ds, hand_ds

{'count': <tf.Tensor: shape=(), dtype=int64, numpy=2366>, '0': <tf.Tensor: shape=(), dtype=int64, numpy=542669205>, '1': <tf.Tensor: shape=(), dtype=int64, numpy=77563499>}

{'count': <tf.Tensor: shape=(), dtype=int64, numpy=1014>, '0': <tf.Tensor: shape=(), dtype=int64, numpy=229833095>, '1': <tf.Tensor: shape=(), dtype=int64, numpy=35980921>}

{'count': <tf.Tensor: shape=(), dtype=int64, numpy=190>, '0': <tf.Tensor: shape=(), dtype=int64, numpy=44178776>, '1': <tf.Tensor: shape=(), dtype=int64, numpy=5628584>}

{'count': <tf.Tensor: shape=(), dtype=int64, numpy=291>, '0': <tf.Tensor: shape=(), dtype=int64, numpy=68873814>, '1': <tf.Tensor: shape=(), dtype=int64, numpy=7410090>}




