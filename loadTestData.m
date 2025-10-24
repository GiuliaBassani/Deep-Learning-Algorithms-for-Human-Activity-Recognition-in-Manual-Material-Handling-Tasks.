function [TestData, TestClass, TestClass_cat, TestT] = loadTestData(sbjfile)
    load(sbj);
    InData = DataBalanced.InData;
    Class = DataBalanced.Class;
    Target = DataBalanced.Target;
    [m,n] = size(InData);
    test_ind = randperm(n);
    % Testing data
    TestData = InData(:,test_ind);
    TestData = TestData';
    TestClass = Class(:,test_ind);
    TestClass = TestClass';
    TestClass_cat = categorical(TestClass);
    TestT = Target(:,test_ind);
end