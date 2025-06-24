classdef combinedCrossEntropyDiceLoss < nnet.layer.ClassificationLayer
    properties
        DiceWeight = [];        % 0 = only CE, 1 = only Dice
        ClassWeights = []        % Per-class weights (vector)
    end

    methods
        function layer = combinedCrossEntropyDiceLoss(name, diceWeight, classWeights)
            % Set properties
            layer.Name = name;
            layer.Description = "Combined Cross-Entropy and Dice Loss with Class Weights";
            layer.DiceWeight = diceWeight;
            layer.ClassWeights = classWeights;
        end

        function loss = forwardLoss(layer, Y, T)
            % Y: predictions [H W C N]
            % T: one-hot ground truth [H W C N]

            % --- Cross-Entropy Loss with Class Weights ---
            W = reshape(layer.ClassWeights, 1, 1, []);  % reshape for broadcasting

            % Avoid log(0)
            logY = log(Y + eps);

            % Weighted CE: - sum over C of (T .* logY .* W)
            weightedCE = -T .* logY .* W;
            ceLoss = mean(weightedCE, 'all');

            % --- Dice Loss ---
            numClasses = size(Y, 3);
            diceLoss = 0;
            smooth = 1;

            for c = 1:numClasses
                probs = Y(:,:,c,:);
                targets = T(:,:,c,:);

                probs = probs(:);
                targets = targets(:);

                intersection = sum(probs .* targets);
                union = sum(probs + targets);

                diceLoss = diceLoss + 1 - (2 * intersection + smooth) / (union + smooth);
            end

            diceLoss = diceLoss / numClasses;

            % --- Combine Losses ---
            loss = (1 - layer.DiceWeight) * ceLoss + layer.DiceWeight * diceLoss;
        end
    end
end
