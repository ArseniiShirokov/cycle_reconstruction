
#!/bin/bash

# Array of timestamp-sequence pairs
pairs=(
    # "2025-09-25_120639 05"
    # "2025-09-25_161454 0b"
    # "2025-09-25_162031 18"
    # "2025-09-25_162538 25"
    "2025-09-25_162900 27"
    "2025-09-25_163049 28"
    "2025-09-25_164212 2f"
    "2025-09-25_164402 3b"
    "2025-09-25_164653 44"
    "2025-09-25_173617 55" 
)

# Loop through each pair and run the command
for pair in "${pairs[@]}"; do
    # Split the pair into timestamp and sequence
    TIMESTAMP=$(echo $pair | cut -d' ' -f1)
    SEQ=$(echo $pair | cut -d' ' -f2)
    
    echo "Running evaluation for TIMESTAMP: $TIMESTAMP, SEQ: $SEQ"
    python nerfstudio/scripts/eval.py --load-config outputs/our_split_av2/splatad/$TIMESTAMP/config.yml --data-root-path /workspace/datasets/self-driving/argoverse2 --output-path splatad_av2_$SEQ.json
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed evaluation for SEQ: $SEQ"
    else
        echo "✗ Failed evaluation for SEQ: $SEQ"
    fi
    echo "----------------------------------------"
done

echo "All evaluations completed!"
