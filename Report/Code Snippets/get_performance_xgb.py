def xgb_predictions(test_batches):
    pred, tgt = model.predict_in_batches(test_batches, filter=filter)

    pred = np.array(pred)
    tgt = np.array(tgt)

    pred = np.squeeze(pred).flatten()
    tgt = np.squeeze(tgt).flatten()

    tgt = tf.convert_to_tensor(tgt, dtype=tf.float32)
    pred = tf.convert_to_tensor(pred, dtype=tf.float32)

    pred_1 = tf.math.equal(pred, tf.ones(shape = pred.shape) )
    pred_0 = tf.math.equal(pred, tf.zeros(shape = pred.shape) )
    tgt_1 = tf.math.equal(tgt, tf.ones(shape = tgt.shape) )
    tgt_0 = tf.math.equal(tgt, tf.zeros(shape = tgt.shape) )

    return pred_1, pred_0, tgt_1, tgt_0

def get_scores(pred_1, pred_0, tgt_1, tgt_0):
    TP = tf.math.count_nonzero( 
        tf.boolean_mask(
            tf.math.equal(pred_1, tgt_1),
            mask = tgt_1
        )
    )

    FP = tf.math.count_nonzero( 
        tf.boolean_mask(
            tf.math.equal(pred_1, tgt_0),
            mask = pred_1
        )
    )
    
    TN = tf.math.count_nonzero( 
        tf.boolean_mask(
            tf.math.equal(pred_0, tgt_0),
            mask = tgt_0
        )
    )

    FN = tf.math.count_nonzero( 
        tf.boolean_mask(
            tf.math.equal(pred_0, tgt_1),
            mask = pred_0
        )
    )
    return TP, FP, TN, FN
