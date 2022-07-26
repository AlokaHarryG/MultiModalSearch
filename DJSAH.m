clear all
load datasets\WIKI.mat
itr = I_tr;
ttr = T_tr;
ltr = L_tr;
[n,c] = size(ltr);
tldl = zeros(c,n);
for k = 1:c
    for i = 1:n
        tldl(k,i)=ltr(i,k)/norm(ltr(i,:),2);
    end
end
% 我比较弱，本代码中的所有矩阵均为转置TT
% calculate Similarity matrix
S = 2*(tldl'*tldl)-ones(n,1)*ones(n,1)';
% 下面是核函数，核函数的b暂时当做目标的哈希码的行值
% Binary Hash Code B:变化值16bits/32bits/64bits/128bits
Kt = {itr,ttr};
B = zeros(n,16);

