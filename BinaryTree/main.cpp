#include <iostream>
#include <iostream>
#include <queue>


#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define new new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
// allocations to be of _CLIENT_BLOCK type
#else
#define DBG_NEW new
#endif


using namespace std;


struct Node {
	int data;
	struct Node* left, * right;
};

struct Node* newNode(int key)
{
	struct Node* temp = new Node;
	temp->data = key;
	temp->left = temp->right = NULL;
	return temp;
};

Node* t = NULL;

struct Trunk
{
	Trunk* prev;
	string str;

	Trunk(Trunk* prev, string str)
	{
		this->prev = prev;
		this->str = str;
	}
};


void deallocate(Node* temp, int key)
{
	if (!temp)
		return;
	deallocate(temp->left, key);
	deallocate(temp->right, key);
	if (temp->data != key)
		delete temp;
}

void perebor(struct Node* temp, int data) {
	if (!temp)
		return;

	if (temp->right != nullptr) {
		if (temp->right->data == data) {
			t = temp->right;
		}
	}
	if (temp->left != nullptr) {
		if (temp->left->data == data) {
			t = temp->left;
		}
	}
	perebor(temp->right, data);
	perebor(temp->left, data);
}

void nullOnKey(struct Node* temp, int data) {
	if (!temp) // if(temp == NULL)
		return;

	if (temp->right != nullptr) {
		if (temp->right->data == data) {
			delete temp->right;
			temp->right = NULL;
		}
	}
	if (temp->left != nullptr) {
		if (temp->left->data == data) {
			delete temp->left;
			temp->left = NULL;
		}
	}
	nullOnKey(temp->right, data);
	nullOnKey(temp->left, data);
}

void del(struct Node* temp, int data) 
{
	if (data == -1)
		return;
	perebor(temp, data);
	deallocate(t, data);
	nullOnKey(temp, data);
}

// Helper function to print branches of the binary tree
void showTrunks(Trunk * p)
{
	if (p == nullptr) {
		return;
	}

	showTrunks(p->prev);
	cout << p->str;
}


void printTree(Node * root, Trunk * prev, bool isLeft)
{
	if (root == nullptr) {
		return;
	}

	string prev_str = "    ";
	Trunk* trunk = new Trunk(prev, prev_str);
	printTree(root->right, trunk, true);

	if (!prev) {
		trunk->str = "———";
	}
	else if (isLeft)
	{
		trunk->str = ".———";
		prev_str = "   |";
	}
	else {
		trunk->str = "`———";
		prev->str = prev_str;
	}

	showTrunks(trunk);
	if (root->data == 0)
		cout << " " << " " << endl;
	else {
		cout << " " << root->data << endl;
	}
	if (prev) {
		prev->str = prev_str;
	}
	trunk->str = "   |";
	printTree(root->left, trunk, false);
	delete trunk;
}

int main()
{
	struct Node* root = newNode(9);

	root->left = newNode(15);

	root->left->left = newNode(3);
	root->left->right = newNode(8);

	root->left->left->left = newNode(6);
	root->left->left->right = newNode(17);

	root->left->right->left = newNode(21);
	root->left->right->right = newNode(1);

	root->right = newNode(5);

	root->right->left = newNode(2);
	root->right->right = newNode(13);

	root->right->left->left = newNode(4);
	root->right->left->right = newNode(11);

	root->right->right->left = newNode(26);
	root->right->right->right = newNode(7);

	root->left->left->left->left = newNode(77);
	root->left->left->left->right = newNode(88);
	root->left->left->right->right = newNode(99);
	root->left->left->right->left = newNode(44);
	root->left->right->left->left = newNode(11);
	root->left->right->left->right = newNode(22);
	root->left->right->right->right = newNode(33);
	root->left->right->right->left = newNode(55);

	printTree(root, nullptr, false);
	cout << endl << endl;
	int key = NULL;
	while (key != -1)
	{
		cout << "Удалить узел № ";
		cin >> key;
		del(root, key);
		printTree(root, nullptr, false);
		cout << endl << endl;
	}
	cout << endl << endl;
	deallocate(root, -1);

	system("pause");
	return 0;
}

