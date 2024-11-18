#ifndef LINKED_LIST_H
#define LINKED_LIST_H

#include <exception>
#include "Entity.h"

// ensure that when a composite entity is created which
// consists of multiple entities, that those entities
// exist in consecutive nodes (in the world's linked list
// of entities). this will faciliate rapid deletions of
// an entire object. also note that each entity in the
// composite entity can itself be a composite entity,
// and those "sub"-entities should also meet the criteria
// of existing in consecutive nodes to facilitate rapid
// entity delitions.

// most deletions will take place during "clean up" (deleting entities marked for deletion)

class EntityList
{
public:
    EntityList()
    {
        head = new Node; // dummy head
        if (head == NULL)
            throw std::exception("Unable to create head node");

        tail = new Node; // dummy tail
        if (tail == NULL)
            throw std::exception("Unable to create tail node");

        head->prev = NULL;
        head->next = tail;

        tail->next = NULL;
        tail->prev = head;

        count = 0;
    };

    EntityList(const EntityList &copy)
    {
        head = new Node; // dummy head
        if (head == NULL)
            throw std::exception("Unable to create head node");

        tail = new Node; // dummy tail
        if (tail == NULL)
            throw std::exception("Unable to create tail node");

        count = 0;
        insert(copy, head);
    };

    void tick()
    {
        for (Node *tmp = head->next; tmp != tail; tmp = tmp->next)
        {
            // for a composite entity, which parts are marked
            // as expired is handled within the composite entity.
            // here, we just delete them.

            // also, since the same object can be pointed to in
            // multiple locations, the Universe class is the only
            // place where actually deleting the entity from memory
            // takes place.
            if (tmp->ent->isExpired())
            {
                tmp->prev->next = tmp->next;
                tmp->next->prev = tmp->prev;
                delete tmp;
                --count;
            }
        }
    };

    unsigned int getCount()
    {
        return count;
    };

    void insert(const EntityList &copy)
    {
        insert(copy, head);
    };

    void insert(Entity *ent)
    {
        insert(ent, head);
    };

private:
    struct Node
    {
        Entity  *ent;
        Node    *next;
        Node    *prev;
    };

    Node *head;
    Node *tail;
    unsigned int count;

    void insert(Entity *ent, Node *afterThis)
    {
        if (afterThis == NULL)
            throw std::exception("Invalid afterThis node specified");

        Node *newNode = new Node;
        if (newNode == NULL)
            throw std::exception("Unable to create new node");

        newNode->ent = ent;

        newNode->next = afterThis->next;
        newNode->prev = afterThis;
        afterThis->next->prev = newNode;
        afterThis->next = newNode;

        ++count;
    };

    // after insert, this list will point to the same instantiated entities that 
    // the other list points to. these entities will be a subset of the entities
    // that this list points to
    //
    // note: copy from tail to head to preserve original ordering of copy in this
    void insert(const EntityList &copy, Node *afterThis)
    {
        if (afterThis == NULL)
            throw std::exception("Invalid afterThis node specified");

        Node *newNode;
        Node *current = copy.tail->prev;
        while (current != copy.head)
        {
            newNode = new Node;
            if (newNode == NULL)
                throw std::exception("Unable to create new node");

            newNode->ent = current->ent; // do not make a copy; point to same instantiated Entity object

            newNode->next = afterThis->next;
            newNode->prev = afterThis;
            afterThis->next->prev = newNode;
            afterThis->next = newNode;

            current = current->prev;
        }

        count += copy.count;
    };
};

#endif