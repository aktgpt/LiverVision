classdef optimizeLine
    %OPTIMIZELINE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ptMat
        initlineVect
    end
    methods(Static)
        function errorFn = errorFunction(lineVect)
            errorFn = mean(distPointsfromLine(this.ptMat, lineVect));
        end
        
        function [c, ceq] = constraint(lineVect)
            c = [];
            ceq = norm([lineVect(4),lineVect(5),lineVect(6)]) - 1;
        end

    end
    
    methods
                function optimizedLine = optimizeLineParameters()
            optimizedLine = fmincon(@errorFunction,this.initlineVect,[],[],[],[],[],[],@constraint);
                end
                function distMat = distPointsfromLine(ptMat,lineVect)
            
            dirVect = [lineVect(4),lineVect(5),lineVect(6)];
            
            for i = 1:length(ptMat)
                ptsVect = ptMat(i,:) - [lineVect(1),lineVect(2),lineVect(3)];
                distMat(i) = norm(norm(cross(ptsVect, dirVect))/norm(dirVect));
            end
            
        end

        
    end
end