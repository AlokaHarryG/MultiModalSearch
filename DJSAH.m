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
% �ұȽ������������е����о����Ϊת��TT
% calculate Similarity matrix
S = 2*(tldl'*tldl)-ones(n,1)*ones(n,1)';
% �����Ǻ˺������˺�����b��ʱ����Ŀ��Ĺ�ϣ�����ֵ
% Binary Hash Code B:�仯ֵ16bits/32bits/64bits/128bits
Kt = {itr,ttr};
B = zeros(n,16);

