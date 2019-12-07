#!/usr/bin/env bash


function run_learner() {
    printf "Loss Type: ${LOSS}\n"
    printf "|P|: ${SIZE_P}\n"
    printf "|N|: ${SIZE_N}\n"
    printf "|U|: ${SIZE_U}\n"
    printf "Pos: ${POS}\n"
    printf "Neg: ${NEG}\n"
    printf "Learning Rate: ${LR}\n"
    printf "Batch Size: ${BATCH_SIZE}\n"

    log_enabled_disabled "Preprocessed" "${PREPROCESS}"
    # log_enabled_disabled "Bias" "${BIAS}"
    log_enabled_disabled "Rho" "${RHO}"
    log_enabled_disabled "Gamma" "${GAMMA}"
    log_enabled_disabled "Tau" "${TAU}"
    printf "Bias: ${BIAS_FLAG}\n"

    BIAS_FLAG=$(construct_cli_string "bias" "${BIAS}")
    GAMMA_FLAG=$(construct_cli_string "gamma" "${GAMMA}")
    if [[ ${LOSS} == "pubn" ]]; then
        TAU_FLAG=$(construct_cli_string "tau" "${TAU}")
        RHO_FLAG=$(construct_cli_string "rho" "${RHO}")
    else
        TAU_FLAG=()
        RHO_FLAG=()
    fi

    python3 ${DRIVER_DIR}/driver.py ${SIZE_P} ${SIZE_N} ${SIZE_U} ${LOSS} \
                                    --pos ${POS} --neg ${NEG} \
                                    --bs ${BATCH_SIZE} --lr ${LR} \
                                    ${TAU_FLAG} ${RHO_FLAG} ${GAMMA_FLAG} \
                                    ${PREPROCESS} ${BIAS_FLAG}
}


function log_enabled_disabled() {
    if [ $# -gt 2 ]; then
        printf "Invalid input count for variable state logged. Exiting...\n"
        exit 1
    fi
    VAR_NAME=$1
    if [ $# -eq 2 ]; then
        VAR_VAL=$2
    else
        VAR_VAL=""
    fi
    if [ -z ${VAR_VAL} ]; then
        STATE=Disabled
        VAR_STR=""
    else
        STATE=Enabled
        VAR_STR="(${VAR_VAL})"
    fi
    printf "${VAR_NAME}: ${STATE} ${VAR_STR}\n"
}


function construct_cli_string() {
    if [ $# -ne 2 ]; then
        printf "Invalid number of arguments to \"construct_cli_string\"\n"
        exit 1
    fi
    FLAG_NAME=$1
    FLAG_VAL=$2
    if [ -z "${FLAG_VAL}" ]; then
        echo ""
    else
        echo "--${FLAG_NAME} ${FLAG_VAL}"
    fi
}

SCRIPT_DIR=$( realpath "$(dirname "$0")"/ )  # Directory containing all sub folders
DRIVER_DIR="${SCRIPT_DIR}/.."

POS="alt comp misc rec"
NEG="sci soc talk"
SIZE_P=500
SIZE_N=500
SIZE_U=6000

BATCH_SIZE=250

PREPROCESS="--preprocess"

MAX_NUM_ITER=10
for itr in $(seq 1 ${MAX_NUM_ITER}); do
    printf "Starting iteration ${itr} of ${MAX_NUM_ITER}...\n"
    LR_ARR=( "1E-3" "5E-4" "5E-3" )
    for LR in "${LR_ARR[@]}"; do
        # Try with no bias as a baseline
        BIAS=""
        RHO=""
        LOSS="pn"
        run_learner

        # Biased set
        BIAS_ARR=( "1 0 0" "0 0 1" "0.1 0.5 0.4" )
        RHO_ARR=( 0.21 0.17 0.1 )
        for ((i=0;i<${#BIAS_ARR[@]};++i)); do
            BIAS="${BIAS_ARR[i]}"
            RHO="${RHO_ARR[i]}"
            LOSS="pn"
            run_learner
            GAMMA_ARR=( 0.1 0.3 0.5 0.7 0.9 1.0 )
            for GAMMA in "${GAMMA_ARR[@]}"; do
                LOSS="nnpu"
                run_learner

                # Tau only affects PUBN
                LOSS="pubn"
                TAU_ARR=( 0.7 0.5 0.9 )
                for TAU in "${TAU_ARR[@]}"; do
                    run_learner
                done
            done
        done
    done
    # New dataset split
    PATH_TO_DELETE="${HOME}/projects/nlp/tensors"
    printf "Deleting folder: ${PATH_TO_DELETE}\n"
    rm -rf "${PATH_TO_DELETE}" > /dev/null
done
