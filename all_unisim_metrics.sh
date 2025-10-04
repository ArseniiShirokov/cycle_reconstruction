
#!/bin/bash

# Array of timestamp-sequence pairs
pairs=(
    # "2025-09-27_171110 05"
    # "2025-09-27_171132 28"
    # "2025-09-27_184543 0b"
    # "2025-09-27_184556 2f"
    # "2025-09-27_195455 18"
    # "2025-09-27_201716 3b"
    # "2025-09-27_205627 25"
    # "2025-09-27_214812 44"
    # "2025-09-27_215621 27"
    "2025-09-30_215841 55"
)

# Loop through each pair and run the command
for pair in "${pairs[@]}"; do
    # Split the pair into timestamp and sequence
    TIMESTAMP=$(echo $pair | cut -d' ' -f1)
    SEQ=$(echo $pair | cut -d' ' -f2)
    
    echo "Running evaluation for TIMESTAMP: $TIMESTAMP, SEQ: $SEQ"
    python nerfstudio/scripts/eval.py --load-config outputs/our_split_av2/unisim/$TIMESTAMP/config.yml --data-root-path /workspace/datasets/self-driving/argoverse2 --output-path unisim_av2_$SEQ.json
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed evaluation for SEQ: $SEQ"
    else
        echo "✗ Failed evaluation for SEQ: $SEQ"
    fi
    echo "----------------------------------------"
done

echo "All evaluations completed!"
