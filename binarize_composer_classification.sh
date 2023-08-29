

PREFIX=$1
[[ -z "$PREFIX" ]] && { echo "PREFIX is empty" ; exit 1; }
[[ -d "${PREFIX}_data_bin" ]] && { echo "output directory ${PREFIX}_data_bin already exists" ; exit 1; }

DATA_RAW=${PREFIX}_data_raw
DATA_BIN=${PREFIX}_data_bin

WORKERS=$2
if [[ -z "${WORKERS}" ]]
then
  WORKERS=24
fi

echo Number of workers: ${WORKERS}

set -x
fairseq-preprocess \
    --only-source \
    --trainpref ${PREFIX}_data_raw/midi_train.txt \
    --validpref ${PREFIX}_data_raw/midi_valid.txt \
    --testpref ${PREFIX}_data_raw/midi_test.txt \
    --destdir ${PREFIX}_data_bin/input0 \
    --srcdict ${PREFIX}_data_raw/dict.input.txt \
    --workers $WORKERS
# I'm not sure we actually need to binarize the targets given we are copying the raw
# labels below.
fairseq-preprocess \
    --only-source \
    --trainpref ${PREFIX}_data_raw/targets_train.txt \
    --validpref ${PREFIX}_data_raw/targets_valid.txt \
    --testpref ${PREFIX}_data_raw/targets_test.txt \
    --destdir ${PREFIX}_data_bin/label \
    --srcdict ${PREFIX}_data_raw/dict.targets.txt \
    --workers $WORKERS
# When we read the binarized version, there seems to be an extra token (EOS?) added to
# each row.
cp ${PREFIX}_data_raw/$i/targets_train.txt ${PREFIX}_data_bin/$i/label/train.label
cp ${PREFIX}_data_raw/$i/targets_valid.txt ${PREFIX}_data_bin/$i/label/valid.label
cp ${PREFIX}_data_raw/$i/targets_test.txt ${PREFIX}_data_bin/$i/label/test.label
set +x

# fairseq-preprocess \
#   --trainpref ~/tmp/composer_classification_data_raw/renamed/train \
#   --validpref ~/tmp/composer_classification_data_raw/renamed/valid \
#   --testpref ~/tmp/composer_classification_data_raw/renamed/test \
#   --source-lang input --target-lang label \
#   --destdir ~/tmp/composer_class2_bin
