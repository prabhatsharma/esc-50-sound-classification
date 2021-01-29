
#!/bin/bash

eksctl create iamserviceaccount \
    --name audio-sa \
    --namespace audio \
    --cluster arm5 \
    --attach-policy-arn arn:aws:iam::525158249545:policy/audio-policy \
    --approve \
    --override-existing-serviceaccounts