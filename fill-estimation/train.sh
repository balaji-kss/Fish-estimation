LOGFILE=loggers/${1}.log

python3 train.py > "$LOGFILE" 2>&1 &

#python3 train_tenc_recon.py > "$LOGFILE" 2>&1 &