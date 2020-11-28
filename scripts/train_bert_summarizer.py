# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.random.set_seed(100)
#tf.config.optimizer.set_jit(True)
import time
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
from preprocess import create_train_data, map_batch_shuffle
from preprocess import *
from hyper_parameters import h_parms
from configuration import config
from metrics import optimizer, loss_function, label_smoothing, get_loss_and_accuracy, tf_write_summary, monitor_run
from input_path import file_path
from creates import log, train_summary_writer, valid_summary_writer
from create_tokenizer import tokenizer, model
from local_tf_ops import *
from numpy.random import randint
from pythonrouge.pythonrouge import Pythonrouge
import tensorflow_datasets as tfds
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import re, string
import numpy as np

#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)
#optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

tf.debugging.set_log_device_placement(True)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    #@tf.function(input_signature=train_step_signature)
    @tf.function
    def train_step(inputs):
      input_ids, input_mask, input_segment_ids, target_ids_, target_mask, target_segment_ids, target_ids, draft_mask, refine_mask, grad_accum_flag = inputs
      with tf.GradientTape() as tape:

        (draft_predictions, draft_attention_weights,
          refine_predictions, refine_attention_weights) = model(
                                                               input_ids, input_mask, input_segment_ids,
                                                               target_ids_, target_mask, target_segment_ids,
                                                               True
                                                                   )
        train_variables = model.trainable_variables
        draft_summary_loss = loss_function(target_ids[:, 1:, :], draft_predictions, draft_mask)
        refine_summary_loss = loss_function(target_ids[:, :-1, :], refine_predictions, refine_mask)
        loss = draft_summary_loss + refine_summary_loss
        loss = tf.reduce_mean(loss)
        #loss = optimizer.get_scaled_loss(loss)
      gradients  = tape.gradient(loss, train_variables)
      #gradients = optimizer.get_unscaled_gradients(gradients)
      # Initialize the shadow variables with same type as the gradients
      if not accumulators:
        for tv in gradients:
          accumulators.append(tf.Variable(tf.zeros_like(tv), trainable=False))
      # accmulate the gradients to the shadow variables
      for (accumulator, grad) in zip(accumulators, gradients):
        accumulator.assign_add(grad)
      # apply the gradients and reset them to zero if the flag is true

      if grad_accum_flag:
        optimizer.apply_gradients(zip(accumulators, train_variables))
        for accumulator in (accumulators):
            accumulator.assign(tf.zeros_like(accumulator))

        train_loss(loss)
        train_accuracy(target_ids_[:, :-1], refine_predictions)
      return (loss,target_ids_[:, :-1], refine_predictions)

    @tf.function(input_signature=val_step_signature)
    def val_step(input_ids,
                 input_mask,
                 input_segment_ids,
                 target_ids_,
                 target_mask,
                 target_segment_ids,
                 target_ids,
                 draft_mask,
                 refine_mask,
                 step,
                 create_summ):
      (draft_predictions, draft_attention_weights,
       refine_predictions, refine_attention_weights) = model(
                                                             input_ids, input_mask, input_segment_ids,
                                                             target_ids_, target_mask, target_segment_ids,
                                                             False
                                                             )
      draft_summary_loss = loss_function(target_ids[:, 1:, :], draft_predictions, draft_mask)
      refine_summary_loss = loss_function(target_ids[:, :-1, :], refine_predictions, refine_mask)
      loss = draft_summary_loss + refine_summary_loss
      loss = tf.reduce_mean(loss)
      validation_loss(loss)
      validation_accuracy(target_ids_[:, :-1], refine_predictions)
      if create_summ:
        rouge, bert = tf_write_summary(target_ids_[:, :-1], refine_predictions, step)
      else:
        rouge, bert = (1.0, 1.0)
      return (rouge, bert)

    def check_ckpt(checkpoint_path):
        ckpt = tf.train.Checkpoint(
                                   model=model,
                                   optimizer=optimizer
                                  )
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
        if tf.train.latest_checkpoint(checkpoint_path):
          ckpt.restore(ckpt_manager.latest_checkpoint)
          log.info(ckpt_manager.latest_checkpoint +' restored')
          latest_ckpt = int(ckpt_manager.latest_checkpoint[-2:])
        else:
            latest_ckpt=0
            log.info('Training from scratch')
        return (ckpt_manager, latest_ckpt)


    @tf.function
    def distributed_train_step(dist_inputs):
        per_replica_losses,target_ids_ , refine_predictions = strategy.run(train_step, args=(dist_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None),target_ids_ , refine_predictions


    train_dataset, val_dataset, num_of_train_examples, _ = create_train_data()
    val_dataset = strategy.experimental_distribute_dataset(val_dataset)
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    train_loss, train_accuracy = get_loss_and_accuracy()
    validation_loss, validation_accuracy = get_loss_and_accuracy()
    accumulators = []
    # if a checkpoint exists, restore the latest checkpoint.
    ck_pt_mgr, latest_ckpt = check_ckpt(file_path.checkpoint_path)
    total_steps = int(h_parms.epochs * (h_parms.accumulation_steps))
    #train_dataset = train_dataset.repeat(total_steps)
    count=0
    ds_train_size = 287113
    length = 1
    start = randint(ds_train_size - length, size=1)[0]
    train_buffer_size = 287113
    for (step, (input_ids, input_mask, input_segment_ids, target_ids_, target_mask, target_segment_ids)) in enumerate(train_dataset):
        #if step >= config.start_from_batch:
        try:
            sum_hyp = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(input_ids) if i not in [CLS_ID, SEP_ID, 0]])
            ip_ids = tokenizer.encode(' '.join(sum_hyp))
            if len(input_ids[0]) >= 512:
                start = randint(ds_train_size - length, size=1)[0]
                examples, metadata = tfds.load('cnn_dailymail', with_info=True, as_supervised=True,
                                               data_dir='/content/drive/My Drive/Text_summarization/cnn_dataset',
                                               builder_kwargs={"version": "3.0.0"},
                                               split=tfds.core.ReadInstruction('train', from_=start, to=start + length,
                                                                               unit='abs'))
                train_examples = examples
                train_dataset = map_batch_shuffle(train_examples, train_buffer_size, split='train', shuffle=True,
                                                  batch_size=1, filter_off=False)
                repeat = True
                continue
            # if len(input_ids) >= 512 or len(input_mask)>=512 or len(input_segment_ids)>=512 or len(ip_ids)>=512:
            #     print("Too much long!!!")
            #     continue
            #     # while len(ip_ids) >= 512:
            #     #     start = randint(ds_train_size - length, size=1)[0]
            #     #     examples, metadata = tfds.load('cnn_dailymail', with_info=True, as_supervised=True,
            #     #                                  data_dir='/content/drive/My Drive/Text_summarization/cnn_dataset',
            #     #                                  builder_kwargs={"version": "3.0.0"},
            #     #                                  split=tfds.core.ReadInstruction('train', from_=start, to=start + length,
            #     #                                                                  unit='abs'))
            #     #     train_examples = examples
            #     #     train_dataset = map_batch_shuffle(train_examples, train_buffer_size, split='train', shuffle=True, batch_size=1, filter_off=False)
            #     #     (input_ids, input_mask, input_segment_ids, target_ids_, target_mask, target_segment_ids) =iter(train_dataset)
            # if len(target_ids_) >= 512 or len(target_mask)>512 or len(target_segment_ids)>512:
            #     print('maggoty bread')
            #     continue

            print(np.where(input_mask.numpy()[0])[0].max())
            print(len(input_ids[0]))
            count+=1
            start=time.time()
            draft_mask = tf.math.logical_not(tf.math.equal(target_ids_[:, 1:], 0))
            refine_mask = tf.math.logical_not(tf.math.equal(target_ids_[:, :-1], 0))
            target_ids = label_smoothing(tf.one_hot(target_ids_, depth=config.input_vocab_size))
            grad_accum_flag = True if (step+1)%h_parms.accumulation_steps == 0 else False
            #target_x, refine_predictions=train_step(input_ids,input_mask,input_segment_ids, target_ids_,target_mask,target_segment_ids,target_ids,draft_mask,refine_mask, grad_accum_flag)
            inputs=[input_ids,input_mask,input_segment_ids, target_ids_,target_mask,target_segment_ids,target_ids,draft_mask,refine_mask, grad_accum_flag]
            target_x, refine_predictions=distributed_train_step(inputs)

            if grad_accum_flag:
              batch_run_check(
                            step+1,
                            start,
                            train_summary_writer,
                            train_loss.result(),
                            train_accuracy.result(),
                            model
                            )
            eval_frequency = ((step+1) * h_parms.batch_size) % config.eval_after
            print('end if iter')
            if eval_frequency == 0:
              predicted = (tokenizer.decode([i for i in tf.squeeze(tf.argmax(refine_predictions,axis=-1)) if i not in [101,102,0]]))
              target = (tokenizer.decode([i for i in tf.squeeze(target_x) if i not in [101,102,0]]))
              print(f'the golden summary is {target}')
              print(f'the predicted summary is {predicted if predicted else "EMPTY"}')
              ckpt_save_path = ck_pt_mgr.save()
              (val_acc, val_loss, rouge_score, bert_score) = calc_validation_loss(
                                                                                  val_dataset,
                                                                                  step+1,
                                                                                  val_step,
                                                                                  valid_summary_writer,
                                                                                  validation_loss,
                                                                                  validation_accuracy
                                                                                  )

              latest_ckpt+=(step+1)
              log.info(
                       model_metrics.format(
                                            step+1,
                                            train_loss.result(),
                                            train_accuracy.result(),
                                            val_loss,
                                            val_acc,
                                            rouge_score,
                                            bert_score
                                           )
                      )
              log.info(evaluation_step.format(step+1, time.time() - start))
              log.info(checkpoint_details.format(step+1, ckpt_save_path))

              #Print metrics:
              pattern = re.compile('[\W_]+')
              infer_ckpt = '75'
              ckpt = tf.train.Checkpoint(model=model)
              ckpt.restore(
                  'ckpt_dir/content/drive/My Drive/Text_summarization/BERT_text_summarisation/Summarization_inference_ckps/ckpt-' + infer_ckpt).expect_partial()

              train_examples = examples['train']
              train_dataset = map_batch_shuffle(train_examples, 100, split='train', shuffle=True, batch_size=1,
                                                filter_off=False)
              for (step, (input_ids, input_mask, input_segment_ids, target_ids_, target_mask, target_segment_ids)) in enumerate(
                      train_dataset):
                  sum_hyp = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(input_ids) if i not in [CLS_ID, SEP_ID, 0]])
                  ip_ids = tokenizer.encode(' '.join(sum_hyp))
                  preds_draft_summary, preds_refined_summary, refine_attention_dist = predict_using_beam_search(
                      tf.convert_to_tensor([ip_ids]),
                      refine_decoder_sampling_type='topktopp',
                      k=7,
                      p=0.8)
                  reference = tokenizer.convert_ids_to_tokens(
                      [i for i in tf.squeeze(target_ids_) if i not in [CLS_ID, SEP_ID, 0]])
                  reference = ' '.join(list(reference))
                  sum_hyp = tokenizer.convert_ids_to_tokens(
                      [i for i in tf.squeeze(preds_refined_summary) if i not in [CLS_ID, SEP_ID, 0]])
                  summary = convert_wordpiece_to_words(sum_hyp)
                  pattern = re.compile("[^A-Za-z ]")
                  summary = pattern.sub('', summary)
                  reference = pattern.sub('', reference)
                  pattern = re.compile("[ ]+")
                  summary = pattern.sub(' ', summary)
                  reference = pattern.sub(' ', reference)
                  rouge = Pythonrouge(summary_file_exist=False,
                                      summary=[[summary]], reference=[[[reference]]],
                                      n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                                      recall_only=True, stemming=True, stopwords=True,
                                      word_level=True, length_limit=True, length=50,
                                      use_cf=False, cf=95, scoring_formula='average',
                                      resampling=True, samples=1000, favor=True, p=0.5)
                  score = rouge.calc_score()
                  print(score)
                  score_avg = (score['ROUGE-1'] + score['ROUGE-2'] + score['ROUGE-L']) / 3
                  print(score_avg)
        except Exception as e:
            print(e)