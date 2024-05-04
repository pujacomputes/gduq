GOODCONFIGSPATH=/p/lustre3/trivedi1/GOOD/configs/final_configs
SRCPATH=/p/lustre3/trivedi1/GDUQ/src/
cd $SRCPATH 

#Dataset Parameters
DATASET=GOODSST2
SHIFTTYPE=covariate #concept
DOMAIN=length

# DATASET=GOODMotif
# SHIFTTYPE=covariate #concept
# DOMAIN=size

# DATASET=GOODCMNIST
# SHIFTTYPE=covariate #concept
# DOMAIN=color


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

python gduq_graphclassification.py --config_path ${GOODCONFIGSPATH}/${DATASET}/${DOMAIN}/${SHIFTTYPE}/ERM.yaml \
    --anchor_type $ANCHORTYPE \
    --gduq_type $GDUQTYPE\
    --layerwise_duq $LAYERWISEDUQ\
    --num_anchors $NUMANCHORS \
    --random_seed $SEED \
    --ood_alg erm \
    --max_epoch 3 \
    --save_path  $SAVEPATH 

cd eval_posthoc

GDUQTYPE=input
ANCHORTYPE=node
UQNAME=ETS

for UQNAME in VS ETS IRM Dirichlet Spline Orderinvariant 
do
CKPT=${SAVEPATH}/${DATASET}/gduq_GOODSST2_length_covariate_vGIN_node-10_-1_input_0.ckpt
CUDA_LAUNCH_BLOCKING=1 python eval_graphclassification_gduq.py --config_path ${GOODCONFIGSPATH}/${DATASET}/${DOMAIN}/${SHIFTTYPE}/ERM.yaml \
    --random_seed ${SEED} \
    --anchor_type $ANCHORTYPE \
    --num_anchors 10 \
    --uq_name $UQNAME \
    --gduq_type $GDUQTYPE\
    --layerwise_duq $LAYERWISEDUQ\
    --ckpt_path $CKPT 
done
