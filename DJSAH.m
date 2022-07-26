clear all
load datasets\WIKI.mat
itr = I_tr;
ttr = T_tr;
ltr = L_tr;
L = full(ind2vec(ltr'));%Labelת�ɶ��Ⱦ���
[c,n] = size(L);
tldl = zeros(c,n);
[~,di] = size(I_tr);
[~,dt] = size(T_tr);
for k = 1:c
    for i = 1:n
        tldl(k,i)=L(k,i)/norm(L(:,i),2);
    end
end
% �ұȽ������������е����о����Ϊת��TT
% calculate Similarity matrix
S = 2*(tldl'*tldl)-ones(n,1)*ones(n,1)';
% �����Ǻ˺������˺�����b��ʱ����Ŀ��Ĺ�ϣ�����ֵ
% Binary Hash Code B:�仯ֵ16bits/32bits/64bits/128bits
Kt = {itr,ttr};
B = zeros(n,16);

