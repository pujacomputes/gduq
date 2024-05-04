GOODCONFIGSPATH=/p/lustre3/trivedi1/GOOD/configs/final_configs
SRCPATH=/p/lustre3/trivedi1/GDUQ/src/
cd $SRCPATH 

#Dataset Parameters
DATASET=GOODCBAS
SHIFTTYPE=covariate #concept
DOMAIN=color


DATASET=GOODCora
SHIFTTYPE=covariate #concept
DOMAIN=word

#GDUQ Parameters
GDUQTYPE=input
ANCHORTYPE=node
NUMANCHORS=10
LAYERWISEDUQ=-1

#GDUQ Parameters
# GDUQTYPE=hiddenreps
# ANCHORTYPE=batch
# NUMANCHORS=10
# LAYERWISEDUQ=-1


#GDUQ Parameters
# GDUQTYPE=layerwise
# ANCHORTYPE=random
# NUMANCHORS=10
# LAYERWISEDUQ=2


#Bookkeeping Parameters
SEED=0
SAVEPATH=/p/lustre3/trivedi1/GDUQ/ckpts

echo "${GOODCONFIGSPATH}/${DATASET}/${DOMAIN}/${SHIFTTYPE}/"
echo "Dataset: ${DATASET}"
echo "Shifttype: ${SHIFTTYPE}"
echo "Domain: ${DOMAIN}"
echo "AnchorType: ${ANCHORTYPE}"
echo "Seed: ${SEED}"

python gduq_nodeclassification.py --config_path ${GOODCONFIGSPATH}/${DATASET}/${DOMAIN}/${SHIFTTYPE}/ERM.yaml \
    --anchor_type $ANCHORTYPE \
    --gduq_type $GDUQTYPE\
    --layerwise_duq $LAYERWISEDUQ\
    --num_anchors $NUMANCHORS \
    --random_seed $SEED \
    --ood_alg erm \
    --max_epoch 2 \
    --save_path  $SAVEPATH 


for UQNAME in VS ETS IRM Dirichlet Spline Orderinvariant 
do
cd ${SRCPATH}/eval_posthoc
CKPT=${SAVEPATH}/GOODCora/gduq_GOODCora_word_covariate_GCN_node-10_-1_input_0.ckpt
python eval_nodeclassification_gduq.py --config_path ${GOODCONFIGSPATH}/${DATASET}/${DOMAIN}/${SHIFTTYPE}/ERM.yaml \
    --anchor_type $ANCHORTYPE \
    --gduq_type $GDUQTYPE\
    --layerwise_duq $LAYERWISEDUQ\
    --num_anchors $NUMANCHORS \
    --random_seed $SEED \
    --ood_alg erm \
    --max_epoch 2 \
    --save_path  $SAVEPATH \
    --uq_name $UQNAME \
    --ckpt_path $CKPT 
done