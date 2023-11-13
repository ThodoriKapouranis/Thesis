@tf.function
def calculate_metrics(img: tf.Tensor, tgt: tf.Tensor, wgt: tf.Tensor):
    """Calculate metrics. Model is defined globally outside of this scope as "model".
       This is intended to be used with Tensorflow's Dataset.map
        
        Example:
        metrics = ds_to_use.map( lambda x, y, z: tf.numpy_function(func=calculate_metrics, inp=[x, y, z], Tout=(tf.int64, tf.int64, tf.int64, tf.int64)) )
        metrics = np.array(list(metrics.as_numpy_iterator()))

    Args:
        img (tf.Tensor): Dataset image
        tgt (tf.Tensor): Dataset target
        wgt (tf.Tensor): Dataset wegiht

    Returns:
        array: [True positive, False Positive, True Negative, False Negative]
    """
    print('...')
    TP, FP, TN, FN = 0, 0, 0, 0
    logits = model(img)
    pred = tf.argmax(logits, axis=3)
    pred = tf.reshape(pred, [-1])
    pred = tf.cast(pred, tf.float32)
    
    tgt = tf.reshape(tgt, [-1])
    tgt =  tf.cast(tgt, tf.float32)

    pred_1 = tf.math.equal(pred, tf.ones(shape = pred.shape) )
    pred_0 = tf.math.equal(pred, tf.zeros(shape = pred.shape) )
    tgt_1 = tf.math.equal(tgt, tf.ones(shape = tgt.shape) )
    tgt_0 = tf.math.equal(tgt, tf.zeros(shape = tgt.shape) )

    print('calculating nonzero - TP')
    TP = tf.math.count_nonzero( 
        tf.boolean_mask(
            tf.math.equal(pred_1, tgt_1),
            mask = tgt_1
        )
    )

    print('calculating nonzero - FP') # Prediction is 1 when target is 0
    FP = tf.math.count_nonzero( 
        tf.boolean_mask(
            tf.math.equal(pred_1, tgt_0),
            mask = pred_1
        )
    )
    
    print('calculating nonzero - TN')
    TN = tf.math.count_nonzero( 
        tf.boolean_mask(
            tf.math.equal(pred_0, tgt_0),
            mask = tgt_0
        )
    )

    print('calculating nonzero - FN')
    FN = tf.math.count_nonzero( 
        tf.boolean_mask(
            tf.math.equal(pred_0, tgt_1),
            mask = pred_0
        )
    )
    return TP, FP, TN , FN