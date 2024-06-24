python3 ../train.py --no_instance \
                    --checkpoints_dir ../checkpoints/ \
                    --name w_attention_unet \
                    --dataroot ../maps/map/ \
                    --saveroot ../maps/w/ \
                    --valroot ../maps/val/ \
                    --gpu_ids 0 \
                    --save_epoch_freq 10 \
                    --loadSize 128 \
                    --crop_and_resize \
                    --randomCrop \
                    --flip \
                    --rotate \
                    --ngf 32 \
                    --lambda_feat 5 \
                    # --no_norm_input
                    # --define_norm \
                    # --saveImage \