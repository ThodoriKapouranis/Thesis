from absl import app, flags
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import xgboost as xgb
from transunet import TransUNet

sys.path.append('../Thesis')
from Models.XGB import Batched_XGBoost
from DatasetHelpers.Preprocessing import PyRAT_rlf, PyRAT_sigma, box_filter, fast_frost_filter, frost_filter, lee_filter
from DatasetHelpers.Dataset import create_dataset, convert_to_tfds

FLAGS = flags.FLAGS
flags.DEFINE_bool("debug", False, "Set logging level to debug")
flags.DEFINE_integer("scenario", 1, "Training data scenario. \n\t 1: Only co_event \n\t 2: coevent & preevent \n\t 3: coevent & preevent & coherence")
flags.DEFINE_string("ds", "hand", "hand or holdout dataset to use for evaluation")
flags.DEFINE_string("model_path", "/workspaces/Thesis/Results/Models/unet_scenario1_64", "'xgboost', 'unet', 'a-unet")
flags.DEFINE_string("model", "NN", " 'xgb' or 'NN' ")
flags.DEFINE_integer('xgb_batches', 12, 'batches to use for splitting xgboost training to fit in memory')
flags.DEFINE_string("filter", None, "None / lee")

flags.DEFINE_string('s1_co', '/workspaces/Thesis/10m_data/s1_co_event_grd', 'filepath of Sentinel-1 coevent data')
flags.DEFINE_string('s1_pre', '/workspaces/Thesis/10m_data/s1_pre_event_grd', 'filepath of Sentinel-1 prevent data')
flags.DEFINE_string('s2_weak', '/workspaces/Thesis/10m_data/s2_labels', 'filepath of S2-weak labelled data')
flags.DEFINE_string('coh_co', '/workspaces/Thesis/10m_data/coherence/co_event', 'filepath of coherence coevent data')
flags.DEFINE_string('coh_pre', '/workspaces/Thesis/10m_data/coherence/pre_event', 'filepath of coherence prevent data')

flags.DEFINE_string('hand_coh_co', '/workspaces/Thesis/10m_hand/coherence_10m/hand_labeled/co_event', '(h) filepath of coevent data')
flags.DEFINE_string('hand_coh_pre', '/workspaces/Thesis/10m_hand/coherence_10m/hand_labeled/pre_event', '(h) filepath of preevent data')
flags.DEFINE_string('hand_s1_co', '/workspaces/Thesis/10m_hand/HandLabeled/S1Hand', '(h) filepath of Sentinel-1 coevent data')
flags.DEFINE_string('hand_s1_pre', '/workspaces/Thesis/10m_hand/S1_Pre_Event_GRD_Hand_Labeled', '(h) filepath of Sentinel-1 prevent data')
flags.DEFINE_string('hand_labels', '/workspaces/Thesis/10m_hand/HandLabeled/LabelHand', 'filepath of hand labelled data')

flags.DEFINE_bool("baseline", False, "T/F for baseline. If true, it does not apply the new processing pipeline")

'''
THIS FILE IS INTENDED TO RUN A PRETRAINED MODEL THROUGH TESTING.

To verify performance of the models, we must test them on
1. The hold out (default: sri-lanka) test dataset (weakly labelled)
2. The hand labelled test dataset.

Performance metrics for each test must include
1. Total mIoU
2. Water mIoU
3. Water Precision
4. Water Recall
5. Water F1
6. Nonwater mIoU
7. Nonwater Precision
8. Nonwater Recall
9. Nonwater F1
'''

'''
IoU = TP / (TP + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
'''

def main(x):
    # USING: Saving whole models so that the architecture does not need to be initialized.
    # IGNORE:  when restoring a model from weights-only, create a model with the same architecture as the original model and then set its weights.
    model = None
    dataset = create_dataset(FLAGS)
    
    def identity(x):
        return x     
    
    filter = identity
    if FLAGS.filter == 'lee':
        filter = lee_filter
    if FLAGS.filter == 'box':
        filter = box_filter
    if FLAGS.filter == 'rfl':
        filter = PyRAT_rlf
    if FLAGS.filter == 'sigma':
        filter = PyRAT_sigma
    if FLAGS.filter == "frost":
        filter = fast_frost_filter
        
    print(filter)

    
    channels = 2
    if FLAGS.scenario == 2:
        channels = 4
    elif FLAGS.scenario == 3:
        channels = 6

    if FLAGS.model == "NN":
        BATCH_SIZE = 1 # For loader to make dimensions not break

        model = tf.keras.models.load_model(FLAGS.model_path)
        print(model.summary())
        _, _, holdout_set, hand_set = convert_to_tfds(dataset, channels, filter=filter)

        holdout_set = (
                holdout_set
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE)
        )

        hand_set = (
                hand_set
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE)
        )
        
        ds_to_use = holdout_set if FLAGS.ds=="holdout" else hand_set

        @tf.function
        def calculate_metrics(img: tf.Tensor, tgt: tf.Tensor, wgt: tf.Tensor):
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
        
        metrics = ds_to_use.map( lambda x, y, z: tf.numpy_function(func=calculate_metrics, inp=[x, y, z], Tout=(tf.int64, tf.int64, tf.int64, tf.int64)) )
        metrics = np.array(list(metrics.as_numpy_iterator()))
        
        print(metrics.shape)
        TP = np.sum(metrics[:,0])
        FP = np.sum(metrics[:,1])
        TN = np.sum(metrics[:,2])
        FN = np.sum(metrics[:,3])
        
        print(TP, FP, TN, FN)
        hand_water_IoU = TP / (TP + FP + FN)
        hand_water_p = TP / (TP + FP)
        hand_water_r = TP / (TP + FN)
        hand_water_f = (2*hand_water_p * hand_water_r) / (hand_water_p + hand_water_r)


        # return  water_IoU, water_p, water_r, water_f

        # hand_water_IoU, hand_water_p, hand_water_r, hand_water_f = calculate_metrics(model, hand_set)
        # # hold_water_IoU, hold_water_p, hold_water_r, hold_water_f = calculate_metrics(model, holdout_set)

        # # print(f'Water IoU:\t\t {100 * hold_water_IoU}:.2f')
        # # print(f'Water Precision:\t\t {100 * hold_water_p}:.2f')
        # # print(f'Water Recall:\t\t {100 * hold_water_r}:.2f')
        # # print(f'Water F1:\t\t {100 * hold_water_f}:.2f')
        # print('\n')
        print(f'Water IoU:\t\t {(100 * hand_water_IoU):.3f}')
        print(f'Water Precision:\t{(100 * hand_water_p):.3f}')
        print(f'Water Recall:\t\t {(100 * hand_water_r):.3f}')
        print(f'Water F1:\t\t {(100 * hand_water_f):.3f}')

    if FLAGS.model == "xgb":
        
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
        
        model = Batched_XGBoost()
        model.load_model(FLAGS.model_path)
        print("Succesfully loaded XGBoost model ...")

        batches = dataset.generate_batches(FLAGS.xgb_batches, which_ds=FLAGS.ds)
        print("Succesfully made batches ...")

        TP, FP, TN, FN = 0, 0, 0, 0

        for k in batches.keys():
            test_batch = {0: batches[k]}
            a, b, c, d = get_scores( *xgb_predictions(test_batch) )
            TP += a
            FP += b
            TN += c
            FN += d

        print(TP, FP, TN, FN)
        hand_water_IoU = TP / (TP + FP + FN)
        hand_water_p = TP / (TP + FP)
        hand_water_r = TP / (TP + FN)
        hand_water_f = (2*hand_water_p * hand_water_r) / (hand_water_p + hand_water_r)
        
        print(f'Water IoU:\t\t {(100 * hand_water_IoU):.3f}')
        print(f'Water Precision:\t{(100 * hand_water_p):.3f}')
        print(f'Water Recall:\t\t {(100 * hand_water_r):.3f}')
        print(f'Water F1:\t\t {(100 * hand_water_f):.3f}')

if __name__ == "__main__":
    app.run(main)
