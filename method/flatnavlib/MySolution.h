#pragma once

#include <vector>
#include <memory>

class Solution {
public:
    // 构造函数和析构函数是必需的
    Solution();
    ~Solution();

    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* res);

private:
    // 向前声明一个我们将在 .cpp 文件中定义的实现类
    class SolutionImpl; 
    
    // 一个指向真正实现的智能指针
    std::unique_ptr<SolutionImpl> pimpl; 
};