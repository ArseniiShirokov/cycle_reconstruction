
#!/bin/bash

# Array of timestamp-sequence pairs
av2_pairs=(
    "2025-09-27_172231 05"
    "2025-09-27_172349 28"
    "2025-09-27_183457 0b"
    "2025-09-27_183514 2f"
    "2025-09-27_194241 18"
    "2025-09-27_195635 3b"
    "2025-09-27_203832 25"
    "2025-09-27_210051 44"
    "2025-09-27_213531 27"
    "2025-09-30_215846 55"
)

# Array of timestamp-sequence pairs
pandaset_pairs=(
    "2025-09-27_172231 05"
    "2025-09-27_172349 28"
    "2025-09-27_183457 0b"
    "2025-09-27_183514 2f"
    "2025-09-27_194241 18"
    "2025-09-27_195635 3b"
    "2025-09-27_203832 25"
    "2025-09-27_210051 44"
    "2025-09-27_213531 27"
    "2025-09-30_215846 55"
)


# Loop through each pair and run the command
for pair in "${av2_pairs[@]}"; do
    # Split the pair into timestamp and sequence
    TIMESTAMP=$(echo $pair | cut -d' ' -f1)
    SEQ=$(echo $pair | cut -d' ' -f2)
    
    echo "Running evaluation for TIMESTAMP: $TIMESTAMP, SEQ: $SEQ"
    python nerfstudio/scripts/eval.py --load-config outputs/our_split_av2/neurad/$TIMESTAMP/config.yml --data-root-path /workspace/datasets/self-driving/argoverse2 --output-path neurad_av2_$SEQ.json
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed evaluation for SEQ: $SEQ"
    else
        echo "✗ Failed evaluation for SEQ: $SEQ"
    fi
    echo "----------------------------------------"
done


# Loop through each pair and run the command
for pair in "${pandaset_pairs[@]}"; do
    # Split the pair into timestamp and sequence
    TIMESTAMP=$(echo $pair | cut -d' ' -f1)
    SEQ=$(echo $pair | cut -d' ' -f2)
    
    echo "Running evaluation for TIMESTAMP: $TIMESTAMP, SEQ: $SEQ"
    python nerfstudio/scripts/eval.py --load-config outputs/our_split/neurad/$TIMESTAMP/config.yml --data-root-path /workspace/datasets/self-driving/argoverse2 --output-path neurad_av2_$SEQ.json
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed evaluation for SEQ: $SEQ"
    else
        echo "✗ Failed evaluation for SEQ: $SEQ"
    fi
    echo "----------------------------------------"
done


echo "All evaluations completed!"
