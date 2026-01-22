multivariateDivision = (h, F) -> (
    R := ring h;
    
    p := h;
    r := 0_R; 
    
    while p != 0 do (
        divisible := false;
        ltP := leadTerm p;
        
        for i from 0 to #F-1 do (
            f := F#i;
            ltF := leadTerm f;
            
            if ltP % ltF == 0 then (
                factor := ltP // ltF;
                p = p - factor * f;
                divisible = true;
                
                break; 
            );
        );
        
        if not divisible then (
            r = r + ltP;
            p = p - ltP;
        );
    );
    
    return r;
);

sPolynomial = (f, g) -> (
    L := lcm(leadMonomial f, leadMonomial g);
    
    multF := L // leadTerm f;
    multG := L // leadTerm g;
    
    return (multF * f) - (multG * g);
);

buchberger = (F) -> (
    G := F;

    pairs := {};
    for i from 0 to #G-1 do (
        for j from i+1 to #G-1 do (
            pairs = append(pairs, {G#i, G#j});
        );
    );
    
    while #pairs > 0 do (
        pair := pairs#0;
        pairs = drop(pairs, 1);
        
        fi := pair#0;
        fj := pair#1;
        
        Spoly := sPolynomial(fi, fj);
        
        r := multivariateDivision(Spoly, G);
        
        if r != 0 then (
            print("Found new generator: " | toString(r));
            
            newPairs := {};
            for g in G do (
                newPairs = append(newPairs, {r, g});
            );
            
            pairs = join(pairs, newPairs);
            G = append(G, r);
        );
    );
    
    return G;
);


minimizeBasis = (G) -> (
    minimalG := {};
    
    for i from 0 to #G-1 do (
        g := G#i;
        isRedundant := false;
        ltG := leadTerm g;
        
        for j from 0 to #G-1 do (
            if i == j then continue; 
            
            f := G#j;
            ltF := leadTerm f;
            
            if ltG % ltF == 0 then (
                isRedundant = true;
                break;
            );
        );
        
        if not isRedundant then (
            minimalG = append(minimalG, g);
        );
    );
    
    return minimalG;
);

makeMonic = (G) -> (
    monicG := {};
    for g in G do (
        lc := leadCoefficient g;
        monicG = append(monicG, (1/lc) * g);
    );
    return monicG;
);


-- Usage for Example 2.3.*
R = QQ[x, y, z, MonomialOrder => GRevLex];
f1 = x^2;
f2 = x*y + y;
f3 = x^2 * y + z;
F = {f1, f2, f3};

Gb = buchberger(F);
GbReduced = minimizeBasis(Gb);
GbReduced = makeMonic(GbReduced);

print "Groebner Basis:";
print Gb;

print "Reduced Groebner Basis:";
print GbReduced;

print "--- M2 Built-in Result ---";
I = ideal F;
print gens gb I;