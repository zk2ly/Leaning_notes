#include <iostream>
#include <cassert>

using namespace std;

template<typename Key, typename Value>
class BST{

private:
    struct Node{
        Key key;
        Value value;
        Node *left;
        Node *right;

        Node(Key key, Value value){
            this->key = key;
            this->value = value;
            this->left = this->right = NULL;
        }
    };

    // 私有变量 一个根节点和一个节点数
    Node *root;
    int count;

public:
    BST(){
        root = NULL;
        count = 0;
    }


    ~BST(){
        destroy(root);
    }

    int size(){
        return count;
    }

    bool isEmpty(){
        return count==0;
    }

    // 在搜索二叉树中插入结点(key, value)
    void insert(Key key, Value value){
        root = insert(root, key, value);
    }

    // 查找二分搜索树中是否有键为key的结点
    bool contain(Key key){
        return contain(root, key);
    }

    // 查找key对应的值value，在之前要用contain函数确认在树中,不存在返回NULL
    Value* search(Key key){
        return search(root, key);
    }

    // 前序遍历
    void preOrder(){
        preOrder(root);
    }

    // 中序遍历
    void inOrder(){
        inOrder(root);
    }

    // 后序遍历
    void postOrder(){
        postOrder(root);
    }

private:
    //在以*node为根的搜索二叉树中插入结点(key,value)
    Node* insert(Node *node, Key key, Value value){
        if(node = NULL){
            count++;
            return new Node(key, value);
        }

        if(node->key == key)
            node->value = value;
        else if(node->key > key)
            insert(node->left, key, value);
        else
            insert(node->right, key, value);

        return node;    
    }

    bool contain(Node *node, Key key){
        if(node = NULL)
            return false;
        
        if(node->key == key)
            return true;
        else if(k < node->key)
            contain(node->left, key);
        else
            contain(node->right, key);
    }

    Value* search(Node* node, Key key){
        if(node == NULL)
            return NULL;

        if(node->key == key)
            return &(node->value);  // 返回的是指针类型
        else if(k < node->key)
            search(node->left, key);
        else
            search(node->right, key);
    }

    void preOrder(Node *node){
        if(node != NULL){
            cout<<node->key>>endl;
            preOrder(node->left);
            preOrder(node->right);
        }
    }

    void inOrder(Node *node){
        if(node != NULL){
            inOrder(node->left);
            cout<<node->key>>endl;
            inOrder(node->right);
        }
    }

    void postOrder(Node *node){
        if(node != NULL){
            postOrder(node->left);
            postOrder(node->right);
            cout<<node->key>>endl;
        }
    }

    void destroy(Node *node){
        if(node != NUll){
            destroy(node->left);
            destroy(node->right);

            delete node;
            count--;
        }
    }
};

int main(){

    return 0;
}