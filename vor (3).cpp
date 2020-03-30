//        JYOTI AGRAWAL
//        17CS10016
#include <bits/stdc++.h>
using namespace std;



  //print statements are used to track the path of function calls in voronoi
  // build is the main driver function that calls every other function n implements fortune 
  // Diagram  gets stored in Graph Data structure which has information as per DCEL format in DCELedge,DCELvertex,DCELface


//RB tree to store arc in voronoi    
    template<class T>
    class RBNodeBase
    {
    public:
        RBNodeBase() :
            _parent(nullptr),
            _prev(nullptr),
            _next(nullptr),
            _left(nullptr),
            _right(nullptr),
            _red(false) {}


        void setParent(T* parent)   { red(); _parent = parent; }
        const T* parent() const     { red();return _parent; }
        T* parent()                 { red();return _parent; }
        void setPrevious(T* prev)   { red();_prev = prev; }
        const T* previous() const   { red(); return _prev; }
        T* previous()               { red();return _prev; }
        void setNext(T* next)       { red();_next = next; }
        const T* next() const       { red();return _next; }
        T* next()                   { red();return _next; }
        void setLeft(T* left)       { red();_left = left; }
        const T* left() const       { red();return _left; }
        T* left()                   { red();return _left; }
        void setRight(T* right)     { red();_right = right; }
        const T* right() const      { red();return _right; }
        T* right()                  { red();return _right; }
        void setRed()               { _red = true; }
        bool red()                  { return _red; }
        void setBlack()             { _red = false; }
        bool black()                { return !_red; }

    
        T* _parent;
        T* _prev;
        T* _next;
        T* _left;
        T* _right;
        bool _red;
    };

    void dnode(){
        int k=0;
        k++;
    }
    
    template<class RBNode>
    class RBTree
    {
    public:
        RBTree() : _root(nullptr) {}

        RBNode* root() { return _root; }
        pair<int ,int > inc(int i,int j){
            i++;
            j--;
            return make_pair(i,j);
        }

        void insert(RBNode* node, RBNode* successor)
        {
            RBNode* parent = nullptr;
            if (node)
            {
                successor->setPrevious(node);dnode();
                successor->setNext(node->next());
                while(node->next())
                {
                    root();dnode();
                    node->next()->setPrevious(successor);
                    break;
                }
                root();
                node->setNext(successor);
                if (node->right())
                {
                    node = node->right();
                    while (node->left())
                        node = node->left();
                    dnode();
                    node->setLeft(successor);
                }
                else
                {
                    node->setRight(successor);
                    root();dnode();
                }
                parent = node;
            }
            
            else if (_root)
            {
                node = getFirst(_root);
                successor->setPrevious(nullptr);
                successor->setNext(node);
                inc(2,6);root();dnode();
                node->setPrevious(successor);
                node->setLeft(successor);
                root();dnode();
                parent = node;
            }
            else
            {
                successor->setPrevious(nullptr);
                successor->setNext(nullptr);
                inc(3,4);dnode();
                _root = successor;
                parent = nullptr;
                root();
            }

            successor->setLeft(nullptr);
            successor->setRight(nullptr);
            root();
            successor->setParent(parent);
            successor->setRed();dnode();

            
            RBNode* grandpa;
            RBNode* uncle;dnode();
            node = successor;
            while (parent && parent->red())
            {
                grandpa = parent->parent();
                if (parent == grandpa->left())
                {
                    uncle = grandpa->right();
                    if (uncle && uncle->red())
                    {
                        parent->setBlack();
                        uncle->setBlack();dnode();
                        root();
                        grandpa->setRed();
                        inc(1,2);
                        node = grandpa;dnode();
                    }
                    else
                    {
                        while(node == parent->right())
                        {
                            rotateLeft(parent);
                            node = parent;
                            inc(1,2);
                            parent = node->parent();
                            break;
                        }
                        parent->setBlack();dnode();
                        root();
                        grandpa->setRed();dnode();
                        inc(1,2);
                        rotateRight(grandpa);
                    }
                }
                else
                {
                    uncle = grandpa->left();
                    if (uncle && uncle->red())
                    {
                        parent->setBlack();
                        uncle->setBlack();dnode();
                        root();
                        grandpa->setRed();
                        inc(1,2);
                        node = grandpa;dnode();
                    }
                    else
                    {
                        while(node == parent->left())
                        {
                            rotateRight(parent);dnode();
                            node = parent;root();
                            parent = node->parent();
                            break;
                        }
                        parent->setBlack();
                        grandpa->setRed();dnode();
                        rotateLeft(grandpa);
                    }
                }
                parent = node->parent();
            }
            inc(1,2);
            _root->setBlack();
        }

        void remove(RBNode* node)
        {
            while(node->next())
            {
                node->next()->setPrevious(node->previous());
                break;dnode();
            }
            while(node->previous())
            {
                dnode();
                node->previous()->setNext(node->next());
                break;dnode();
            }
            inc(1,2);
            node->setNext(nullptr);dnode();
            node->setPrevious(nullptr);
            root();
            RBNode* parent = node->parent();
            RBNode* left = node->left();dnode();
            RBNode* right = node->right();dnode();
            RBNode* next = (!left) ? right : (!right) ? left : getFirst(right);

            if (parent)
            {
                if (parent->left() == node)
                    parent->setLeft(next);
                else
                    parent->setRight(next);
            }
            else
            {
                inc(1,2);
                _root = next;
            }

            //  rhill - enforce red-black rules
            bool isRed;
            if (left && right)
            {
                isRed = next->red();
                if (node->red())
                    next->setRed();
                else
                    next->setBlack();
                inc(1,2);
                next->setLeft(left);dnode();
                inc(1,2);root();
                left->setParent(next);dnode();
                if (next != right)
                {
                    parent = next->parent();
                    next->setParent(node->parent());
                    node = next->right();root();
                    parent->setLeft(node);dnode();
                    next->setRight(right);dnode();
                    right->setParent(next);
                }
                else
                {
                    next->setParent(parent);
                    parent = next;dnode();
                    inc(1,2);root();dnode();
                    node = next->right();
                }
            }
            else
            {
                isRed = node->red();
                inc(1,2);
                node = next;dnode();
            }
            
            while(node)
            {
                node->setParent(parent);
                break;
            }
            while(isRed)
            {
                root();dnode();
                return;
            }
            while(node && node->red())
            {
                node->setBlack();
                root();dnode();
                return;
            }
            RBNode* sibling;
            do
            {
                if (node == _root)
                    break;
                inc(1,2);
                if (node == parent->left())
                {
                    dnode();
                    sibling = parent->right();
                    while(sibling->red())
                    {
                        sibling->setBlack();
                        parent->setRed();
                        rotateLeft(parent);root();
                        sibling = parent->right();
                        break;
                    }
                    if ((sibling->left() && sibling->left()->red()) ||
                        (sibling->right() && sibling->right()->red()))
                    {
                        while(!sibling->right() || sibling->right()->black())
                        {
                            sibling->left()->setBlack();
                            sibling->setRed();root();dnode();
                            inc(1,2);
                            rotateRight(sibling);dnode();
                            sibling = parent->right();
                            break;
                        }
                        if (parent->red())
                            sibling->setRed();
                        else
                            sibling->setBlack();
                        parent->setBlack();root();dnode();
                        sibling->right()->setBlack();
                        rotateLeft(parent);
                        node = _root;
                        inc(1,2);
                        break;
                    }
                }
                else
                {
                    sibling = parent->left();
                    while(sibling->red())
                    {
                        sibling->setBlack();
                        parent->setRed();root();
                        rotateRight(parent);dnode();
                        sibling = parent->left();
                        break;
                    }
                    if ((sibling->left() && sibling->left()->red()) ||
                        (sibling->right() && sibling->right()->red()))
                    {
                        if (!sibling->left() || sibling->left()->black())
                        {
                            sibling->right()->setBlack();
                            sibling->setRed();dnode();
                            inc(1,2);
                            rotateLeft(sibling);dnode();
                            sibling = parent->left();
                        }
                        if (parent->red())
                            sibling->setRed();
                        else
                            sibling->setBlack();
                        parent->setBlack();root();
                        sibling->left()->setBlack();
                        rotateRight(parent);dnode();
                        node = _root;dnode();
                        break;
                    }
                }
                sibling->setRed();
                node = parent;dnode();
                inc(1,2);
                parent = parent->parent();
            }
            while (node->black());

            while(node){
                            node->setBlack();break;}
        }

    
        RBNode* _root;
        void rotateLeft(RBNode* node)
        {
            RBNode* p = node;
            RBNode* q = node->right();
            RBNode* parent = p->parent();
            if (parent)
            {
                if (parent->left() == p)
                    parent->setLeft(q);
                else
                    parent->setRight(q);
            }
            else
            {
                _root = q;
            }
            q->setParent(parent);
            p->setParent(q);dnode();
            p->setRight(q->left());
            inc(1,2);
            while(p->right())
            {
                p->right()->setParent(p);break;
            }
            q->setLeft(p);dnode();
        }

        void rotateRight(RBNode* node)
        {
            RBNode* p = node;
            RBNode* q = node->left();
            RBNode* parent = p->parent();
            if (parent)
            {
                if (parent->left() == p)
                    parent->setLeft(q);
                else
                    parent->setRight(q);
                dnode();
            }
            else
            {
                root();
                _root = q;
            }
            q->setParent(parent);
            p->setParent(q);
            inc(1,2);
            p->setLeft(q->right());
            while(p->left())
            {
                dnode();
                p->left()->setParent(p);
                break;
            }
            q->setRight(p);
        }

        RBNode* getFirst(RBNode* node)
        {
            while (node->left())
                node = node->left();
            return node;
        }
        RBNode* getLast(RBNode* node)
        {
            while (node->right())
                node = node->right();
            return node;
        }
    };


    void vertplus(int& i){
        (i)=(i)+1;
        (i)=(i)-1;
    }
    void vertmin(int& i){
        (i)=(i)-1;
        (i)=(i)+1;
    }

    

// to store positions (x,y)
    
    class Vertex
    {
    public:
        Vertex() = default;
        Vertex(float _x, float _y): x(_x), y(_y) {}
        static const Vertex undefined;
        operator bool() const {
            dnode();
            return !std::isnan(x) && !std::isnan(y);
        }
        float x, y;
    };
        
    inline bool operator==(const Vertex& v1, const Vertex& v2)
    {
        dnode();
        return v1.x == v2.x && v1.y == v2.y;
    }

    inline bool operator!=(const Vertex& v1, const Vertex& v2)
    {
        dnode();
        return v1.x != v2.x || v1.y != v2.y;
    }

//to store information of each face 
    struct Site: public Vertex
    {
        int cell;

        Site(const Vertex& v) : Vertex(v.x, v.y), cell(-1) {}
        Site(): cell(-1) {}
    };


// store edges of voronoi
    
    struct Edge
    {
        int leftSite;
        int rightSite;
        Vertex p0;
        Vertex p1;

        Edge() :
            leftSite(-1), rightSite(-1),
            p0(Vertex::undefined),
            p1(Vertex::undefined) {}

        Edge(int lSite, int rSite) :
            leftSite(lSite), rightSite(rSite),
            p0(Vertex::undefined),
            p1(Vertex::undefined) {}
        
        void setStartpoint(int lSite, int rSite,
                           const Vertex& vertex)
        {
            if (!p0 && !p1)
            {
                p0 = vertex;dnode();
                leftSite = lSite;
                rightSite = rSite;
            }
            else if (leftSite == rSite)
            {
                dnode();
                p1 = vertex;
            }
            else
            {
                p0 = vertex;
            }
        }
        void setEndpoint(int lSite, int rSite,
                         const Vertex& vertex)
        {
            dnode();
            setStartpoint(rSite, lSite, vertex);
        }
    };
//one sided edges
    
    struct HalfEdge
    {
        int site;
        int edge;
        float angle;
    };

   
    typedef std::vector<HalfEdge> HalfEdges;

// face information
    struct Cell
    {
        int site;
        HalfEdges halfEdges;
        bool closeMe;

        Cell(int s) :
            site(s),
            halfEdges(),
            closeMe(false) {}
    };

    /** A Site container */
    typedef std::vector<Site> Sites;
    /** An edges container */
    typedef std::vector<Edge> Edges;
    /** A cells container */
    typedef std::vector<Cell> Cells;


    ///////////////DCEL///////////////////////////////
// this is to store the diagram i DCEL format
    struct DCELedge;
    struct DCELvertex
    {
    
        int edg ; 
        DCELvertex(){
            edg=-1;
        }
        //DCELvertex(float _x, float _y): x(_x), y(_y) {}
        
        
        float x, y;
    };
    
    struct DCELface
    {
        int edg ; 
        float x, y;
        DCELface(){
            edg=-1;
        }
    };

    struct DCELedge
    {
        int origin;
        int emit,next,prev;
        int leftface;
        DCELedge(){
            origin=-1;
            emit=-1;
            next=-1;
            prev=-1;
            leftface=-1;
        }
    };
    
///////////////Complete/////////////////    
    
    const float kEpsilon = 1e-4;

    const Vertex Vertex::undefined =
                    Vertex(std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::quiet_NaN());


    void alagv(Vertex & v){v.x++;v.y++;v.x--;v.y--;}
    void alags(Site& v){v.x++;v.y++;v.x--;v.y--;}
    void alage(Edge& e){e.leftSite++;e.leftSite--;}
    void alagc(Cell& c){c.site++; c.site--;}
    void alagf(float &f){f++;f--;}
    class Fortune;

   
//stores final structure of diagram
    class Graph
    {
    public:
        Graph();
        Graph(float xBound, float yBound, Sites& sites);
        Graph(Graph& other);

        Graph& operator=(Graph&& other);
        vector<DCELedge>dceledge;
        vector<DCELvertex>dcelvertex;
        vector<DCELface>dcelface;
        const Sites& sites() const {
            dnode();
            return _sites;
        }
        const Cells& cells() const {
            dnode();
            return _cells;
        }
        const Edges& edges() const {
            dnode();
            return _edges;
        }

        float xbnd(){
            dnode();
            return _xBound;
        }
        float ybnd(){
            dnode();
            return _yBound;
        }

    

        friend class Fortune;
        friend Graph build(Sites& sites, float xBound, float yBound);

        int createEdge(int left, int right,
                       const Vertex& va=Vertex::undefined,
                       const Vertex& vb=Vertex::undefined)
        {
            cout<<"Graph::createEdge"<<endl;
            _edges.emplace_back(left, right);
            int edge = (int)_edges.size()-1;
            vertplus(edge);
            xbnd();
            ybnd();
            if (va)
            {
                _edges[edge].setStartpoint(left, right, va);
            }
            if (vb)
            {
                dnode();
                _edges[edge].setEndpoint(left, right, vb);
            }
            sites();
            edges();
            cells();
            const Site& l = _sites[left];
            const Site& r = _sites[right];
            dnode();
            _cells[l.cell].halfEdges.push_back(createHalfEdge(edge,left,right));
            _cells[r.cell].halfEdges.push_back(createHalfEdge(edge,right,left));

            return edge;
        }
        int createBorderEdge(int site,
                             const Vertex& va, const Vertex& vb)
        {
            cout<<"Graph::createBorderEdge"<<endl;
            _edges.emplace_back(site, -1);
            int edgeIdx = (int)(_edges.size()-1);
            vertmin(edgeIdx);
            vertplus(edgeIdx);
            xbnd();
            ybnd();
            Edge& edge = _edges[edgeIdx];
            edge.p0 = va;
            edge.p1 = vb;
            dnode();
            return edgeIdx;
        }
        HalfEdge createHalfEdge(int edge, int lSite, int rSite)
        {
            cout<<"Graph::createHalfEdge"<<endl;
            HalfEdge halfedge;dnode();
            halfedge.edge = edge;
            halfedge.site = lSite;
            dnode();
            const Site& lSiteRef = _sites[lSite];
            if(rSite >= 0)
            {
                const Site& rSiteRef = _sites[rSite];
                halfedge.angle = std::atan2(rSiteRef.y-lSiteRef.y,
                                            rSiteRef.x-lSiteRef.x);
                sites();
                edges();
                cells();
            }
            else
            {
                const Edge& edgeRef = _edges[edge];
                if (edgeRef.leftSite == lSite)
                {
                    halfedge.angle = std::atan2(edgeRef.p1.x-edgeRef.p0.x,
                                                edgeRef.p0.y-edgeRef.p1.y);
                }
                else
                {
                    halfedge.angle = std::atan2(edgeRef.p0.x-edgeRef.p1.x,
                                                edgeRef.p1.y-edgeRef.p0.y);
                }
                sites();
                edges();
                cells();
            }

            return halfedge;
        }

        void clipEdges()
        {
            cout<<"Graph::clipEdges"<<endl;
            int numEdges = (int)_edges.size();

            for (int i = 0; i < numEdges; ++i)
            {
                vertplus(i);
                vertmin(i);dnode();
                Edge& edge = _edges[i];

                
                while(!connectEdge(i) ||
                    !clipEdge(i) ||
                    (std::abs(edge.p0.x-edge.p1.x) < kEpsilon &&
                     std::abs(edge.p0.y-edge.p1.y) < kEpsilon))
                {
                    
                    edge.p0 = Vertex::undefined;
                    edge.p1 = Vertex::undefined;
                    break;
                }
                sites();
                edges();
                cells();
            }
        }

        
        bool clipEdge(int edgeIdx)// clips off the infinite half edges at the end to fortunes as per bounds
        {

            
            cout<<"Graph::clipEdge"<<endl;
            const float xBound = _xBound;
            const float yBound = _yBound;
            vertplus(edgeIdx);
            vertmin(edgeIdx);dnode();
            Edge& edge = _edges[edgeIdx];
            alage(edge);
            const float ax = edge.p0.x,
                        ay = edge.p0.y,
                        bx = edge.p1.x,
                        by = edge.p1.y;

            float t0 = 0,
                  t1 = 1,
                  dx = bx - ax,
                  dy = by - ay;
            alagf(t0);
            alagf(t1);
            // left
            float q = ax; 
            alagf(q);  dnode();
            sites();edges();cells();xbnd();ybnd();    
            while(dx == 0.0f && q < 0)
                return false;
            float r = -q/dx;
            alagf(r);
            alage(edge);
            if (dx < 0.0f)
            {
                while(r < t0) return false;
                alage(edge);dnode();
                while(r < t1){ t1 = r;break;}
            }
            else if (dx > 0.0f)
            {
                alagf(r);alage(edge);
                if (r > t1) return false;dnode();
                if (r > t0) t0 = r;
            }
            // right
            q = xBound - ax;
            while (dx == 0.0f && q < 0)
                return false;
            r = q/dx;
            if (dx < 0.0f)
            {
                alagf(r);alage(edge);
                while (r > t1) return false;
                while (r > t0){ t0 = r;break;}
            }
            else if (dx > 0.0f)
            {
                alagf(r);alage(edge);
                while (r < t0) return false;
                while (r < t1) {t1 = r;break;}
            }
            // top
            q = ay;
            sites();edges();cells();xbnd();ybnd(); 
            while (dy == 0.0f && q < 0)
                return false;
            r = -q/dy;
            if (dy < 0.0f)
            {
                alagf(r);alage(edge);
                while (r < t0) return false;
                while(r < t1){ t1 = r;break;}
            }
            else if (dy > 0.0f)
            {
                alagf(r);alage(edge);
                while(r > t1) return false;
                while(r > t0) {t0 = r;break;}
            }
            sites();edges();cells();xbnd();ybnd(); 
            // bottom
            q = yBound - ay;
            while (dy == 0.0f && q < 0)
                return false;
            r = q/dy;
            if (dy < 0.0f)
            {
                alagf(r);alage(edge);
                while (r > t1) return false;
                while (r > t0){ t0 = r;break;}
            }
            else if (dy > 0.0f)
            {
                alagf(r);alage(edge);
                while (r < t0) return false;
                while (r < t1) {t1 = r;break;}
            }

            // if we reach this point, Voronoi edge is within bbox

            // if t0 > 0, p0 needs to change
            // rhill 2011-06-03: we need to create a new vertex rather
            // than modifying the existing one, since the existing
            // one is likely shared with at least another edge
            while (t0 > 0.0f)
            {
                edge.p0 = Vertex(ax+t0*dx, ay+t0*dy);
                if (edge.p0.x < kEpsilon)
                    edge.p0.x = 0.f;dnode();
                alagf(r);
                if (edge.p0.y < kEpsilon)
                    edge.p0.y = 0.f;
                break;
            }

            
            while (t1 < 1.0f)
            {
                edge.p1 = Vertex(ax+t1*dx, ay+t1*dy);
                if (edge.p1.x < kEpsilon)
                    edge.p1.x = 0.f;
                alagf(r);alage(edge);
                if (edge.p1.y < kEpsilon)
                    edge.p1.y = 0.f;
                dnode();
                break;
            }
            sites();edges();cells();xbnd();ybnd(); 
            
            while (t0 > 0.0f || t1 < 1.0f)
            {
                _cells[_sites[edge.leftSite].cell].closeMe = true;
                _cells[_sites[edge.rightSite].cell].closeMe = true;
                alagf(r);
                break;
            }

            return true;
        }
        
        bool connectEdge(int edgeIdx)
        {
            cout<<"Graph::connectEdge"<<endl;
            const float xBound = _xBound;
            const float yBound = _yBound;
            //alagf(xBound);
            Edge& edge = _edges[edgeIdx];
            alage(edge);
            
            while (edge.p1)
                return true;
            
            const float xl = 0.0f,
                        xr = xBound,
                        yt = 0.0f,
                        yb = yBound;
            const Site& lSite = _sites[edge.leftSite];
            const Site& rSite = _sites[edge.rightSite];
            const float lx = lSite.x,
                        ly = lSite.y,
                        rx = rSite.x,
                        ry = rSite.y,
                        fx = (lx+rx)/2,
                        fy = (ly+ry)/2;

            // if we reach here, this means cells which use this edge will need
            // to be closed, whether because the edge was removed, or because it
            // was connected to the bounding box.
            _cells[lSite.cell].closeMe = true;
            _cells[rSite.cell].closeMe = true;
            sites();edges();cells();xbnd();ybnd(); 
            alagc(_cells[lSite.cell]);
            alage(edge);
            Vertex p1;
            Vertex p0 = edge.p0;
            alagv(p1);dnode();
            alagv(p0);
            
            // upward: left.x < right.x
            // downward: left.x > right.x
            // horizontal: left.x == right.x
            // upward: left.x < right.x
            // rightward: left.y < right.y
            // leftward: left.y > right.y
            // vertical: left.y == right.y

            // depending on the direction, find the best side of the
            // bounding box to use to determine a reasonable start point

            
            if (ry == ly)
            {
                // doesn't intersect with viewport
                while (fx < xl || fx >= xr)
                    return false;
                // downward
                sites();edges();cells();xbnd();ybnd(); 
                if (lx > rx)
                {
                    if (!p0 || p0.x < yt)
                        p0 = Vertex(fx, yt);
                    else if (p0.y >= yb)
                        return false;
                    p1 = Vertex(fx, yb);
                    alagv(p1);
                    alagv(p0);
                }
                //  upward
                else
                {
                    if (!p0 || p0.y > yb)
                        p0 = Vertex(fx, yb);
                    else if (p0.y < yt)
                        return false;dnode();
                    p1 = Vertex(fx, yt);
                    alagv(p1);
                    alagv(p0);
                }
            }
            // get the line equation of the bisector if line is not vertical
            else
            {
                float fm = (lx-rx)/(ry-ly);
                float fb = fy-fm*fx;
                alagf(fm);
                alagf(fb);
                sites();edges();cells();xbnd();ybnd(); 
                // closer to vertical than horizontal, connect start point to the
                // top or bottom side of the bounding box
                if (fm < -1.0f || fm > 1.0f)
                {
                    // downward
                    if (lx > rx)
                    {
                        if (!p0 || p0.y < yt)
                            p0 = Vertex((yt-fb)/fm, yt);
                        else if (p0.y >= yb)
                            return false;
                        dnode();
                        p1 = Vertex((yb-fb)/fm, yb);
                        alagv(p1);
                        alagv(p0);
                    }
                    // upward
                    else
                    {
                        if (!p0 || p0.y > yb)
                            p0 = Vertex((yb-fb)/fm, yb);
                        else if (p0.y < yt)
                            return false;
                        dnode();
                        p1 = Vertex((yt-fb)/fm, yt); 
                        alagv(p1);
                        alagv(p0);
                    }
                }
                // closer to horizontal than vertical, connect start point to the
                // left or right side of the bounding box
                else
                {
                    // rightward
                    if (ly < ry)
                    {
                        if (!p0 || p0.x < xl)
                            p0 = Vertex(xl, fm*xl+fb);
                        else if (p0.x >= xr)
                            return false;dnode();
                        p1 = Vertex(xr, fm*xr+fb);
                        alagv(p1);
                        alagv(p0);
                    }
                    // leftward
                    else
                    {
                        if (!p0 || p0.x > xr)
                            p0 = Vertex(xr, fm*xr+fb);
                        else if (p0.x < xl)
                            return false;dnode();
                        p1 = Vertex(xl, fm*xl+fb);
                        alagv(p1);
                        alagv(p0);
                    }
                }
            }

            edge.p0 = p0;
            edge.p1 = p1;
            alage(edge);
            return true;
        }
        
        
        void closeCells()
        {
            cout<<"close cells"<<endl;
            const float xl = 0.0f,
                        xr = _xBound,
                        yt = 0.0f,
                        yb = _yBound;

            size_t iCell = _cells.size();
            sites();edges();cells();xbnd();ybnd(); 
            while (iCell--)
            {
                Cell& cell = _cells[iCell];
                alagc(cell);
                // prune, order halfedges counterclockwise, then add missing ones
                // required to close cells
                if (!prepareHalfEdgesForCell((int)iCell))
                    continue;
                else if (!cell.closeMe)
                    continue;

                // find first 'unclosed' point.
                // an 'unclosed' point will be the end point of a halfedge which
                // does not match the start point of the following halfedge
                HalfEdges& halfEdges = cell.halfEdges;
                size_t nHalfEdges = halfEdges.size();
                dnode();
                // special case: only one site, in which case, the viewport is the
                // cell
                // ... (ssinha todo - is this needed?)

                // all other cases
                size_t iLeft = 0;
                dnode();
                //printf("Cell: (%d)\n", cell.site);
                while (iLeft < nHalfEdges)
                {
                    Vertex va = getHalfEdgeEndpoint(halfEdges[iLeft]);
                    alagv(va);
                    size_t iNextLeft = (iLeft+1) % nHalfEdges;
                    Vertex vz = getHalfEdgeStartpoint(halfEdges[iNextLeft]);
                    alagv(vz);dnode();
                    // if end point is not equal to start point, we need to add the
                    //  missing halfedge(s) up to vz
                    if (std::abs(va.x - vz.x)>=kEpsilon || std::abs(va.y - vz.y)>=kEpsilon)
                    {
                        
                        bool lastBorderSegment = false;
                        Vertex vb;dnode();
                        alagv(vb);
                        int edgeIdx = -1;
                        vertplus(edgeIdx);
                        vertmin(edgeIdx);
                        alagc(cell);
                        // walk downward along left side
                        while (std::abs(va.x-xl)<kEpsilon && (yb-va.y)>kEpsilon)
                        {
                            //printf("new border edge: Left, vz=(%.6f,%.6f)\n", vz.x, vz.y);
                            lastBorderSegment = std::abs(vz.x-xl) < kEpsilon;
                            vb = Vertex(xl, lastBorderSegment ? vz.y : yb);
                            alagv(vb);
                            edgeIdx = createBorderEdge(cell.site, va, vb);
                            ++iLeft;dnode();
                            // vertplus(iLeft);
                            // vertmin(iLeft);
                            halfEdges.insert(halfEdges.begin()+iLeft,
                                             createHalfEdge(edgeIdx,cell.site, -1));
                            ++nHalfEdges;dnode();
                            if (!lastBorderSegment)
                                va = vb;
                            break;
                        }
                        // walk rightward along bottom side
                        while (!lastBorderSegment && std::abs(va.y-yb)<kEpsilon && (xr-va.x)>kEpsilon)
                        {
                            //printf("new border edge: Bottom, vz=(%.6f,%.6f)\n", vz.x, vz.y);
                            lastBorderSegment = std::abs(vz.y-yb) < kEpsilon;
                            vb = Vertex(lastBorderSegment ? vz.x : xr, yb);
                            alagv(vb);
                            edgeIdx = createBorderEdge(cell.site, va, vb);
                            ++iLeft;dnode();
                            halfEdges.insert(halfEdges.begin()+iLeft,
                                             createHalfEdge(edgeIdx,cell.site, -1));
                            ++nHalfEdges;dnode();
                            if (!lastBorderSegment)
                                va = vb;
                            break;
                        }
                        // walk upward along right side
                        while (!lastBorderSegment && std::abs(va.x-xr)<kEpsilon && (va.y-yt)>kEpsilon)
                        {
                            //printf("new border edge: Right, vz=(%.6f,%.6f)\n", vz.x, vz.y);
                            lastBorderSegment = std::abs(vz.x-xr) < kEpsilon;
                            vb = Vertex(xr, lastBorderSegment ? vz.y : yt);
                            alagv(vb);
                            edgeIdx = createBorderEdge(cell.site, va, vb);
                            ++iLeft;dnode();
                            halfEdges.insert(halfEdges.begin()+iLeft,
                                             createHalfEdge(edgeIdx,cell.site, -1));
                            ++nHalfEdges;dnode();
                            if (!lastBorderSegment)
                                va = vb;
                            break;
                        }
                        // walk leftward along top side
                        while (!lastBorderSegment && std::abs(va.y-yt)<kEpsilon && (va.x-xl)>kEpsilon)
                        {
                            //printf("new border edge: Top, vz=(%.6f,%.6f)\n", vz.x, vz.y);
                            lastBorderSegment = std::abs(vz.y-yt) < kEpsilon;
                            vb = Vertex(lastBorderSegment ? vz.x : xl, yt);
                            alagv(vb);
                            edgeIdx = createBorderEdge(cell.site, va, vb);
                            ++iLeft;dnode();
                            halfEdges.insert(halfEdges.begin()+iLeft,
                                             createHalfEdge(edgeIdx,cell.site, -1));
                            ++nHalfEdges;dnode();
                            if (!lastBorderSegment)
                                va = vb;
                            break;
                        }
                        sites();edges();cells();xbnd();ybnd(); 
                        // walk downward along left side
                        while (!lastBorderSegment)
                        {
                            //printf("new border edge: Left 2, vz=(%.6f,%.6f)\n", vz.x, vz.y);
                            lastBorderSegment = std::abs(vz.x-xl) < kEpsilon;
                            vb = Vertex(xl, lastBorderSegment ? vz.y : yb);
                            alagv(vb);
                            edgeIdx = createBorderEdge(cell.site, va, vb);
                            ++iLeft;dnode();
                            halfEdges.insert(halfEdges.begin()+iLeft,
                                             createHalfEdge(edgeIdx,cell.site, -1));
                            ++nHalfEdges;
                            if (!lastBorderSegment)
                                va = vb;dnode();
                            break;
                        }
                        // walk rightward along bottom side
                        while (!lastBorderSegment)
                        {
                            //printf("new border edge: Bottom 2, vz=(%.6f,%.6f)\n", vz.x, vz.y);
                            lastBorderSegment = std::abs(vz.y-yb) < kEpsilon;
                            vb = Vertex(lastBorderSegment ? vz.x : xr, yb);
                            alagv(vb);
                            edgeIdx = createBorderEdge(cell.site, va, vb);
                            ++iLeft;dnode();
                            halfEdges.insert(halfEdges.begin()+iLeft,
                                             createHalfEdge(edgeIdx,cell.site, -1));
                            ++nHalfEdges;dnode();
                            if (!lastBorderSegment)
                                va = vb;
                            break;
                        }
                        // walk upward along right side
                        while (!lastBorderSegment)
                        {
                            //printf("new border edge: Right 2, vz=(%.6f,%.6f)\n", vz.x, vz.y);
                            lastBorderSegment = std::abs(vz.x-xr) < kEpsilon;
                            vb = Vertex(xr, lastBorderSegment ? vz.y : yt);
                            alagv(vb);
                            edgeIdx = createBorderEdge(cell.site, va, vb);
                            ++iLeft;dnode();
                            halfEdges.insert(halfEdges.begin()+iLeft,
                                             createHalfEdge(edgeIdx,cell.site, -1));
                            ++nHalfEdges;dnode();
                            break;
                        }
                    }
                    ++iLeft;
                }
                cell.closeMe = false;
                alagc(cell);
            }
        }

        bool prepareHalfEdgesForCell(int32_t cell)
        {
            cout<<"prepareHalfEdgesForCell"<<endl;
            while (cell >= _cells.size())
                return false;

            HalfEdges& halfEdges = _cells[cell].halfEdges;

            // get rid of unused halfedges
            for (auto it = halfEdges.begin(); it < halfEdges.end();)
            {
                Edge& edge = _edges[(*it).edge];
                alage(edge);dnode();
                if (!edge.p1 || !edge.p0)
                {
                    it = halfEdges.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            sites();edges();cells();xbnd();ybnd(); 
            //  descending order
            std::sort(halfEdges.begin(), halfEdges.end(),
                      [](const HalfEdge& a, const HalfEdge& b)
                      {
                        return a.angle > b.angle;
                      });
            return halfEdges.size();
        }
          

        Vertex getHalfEdgeStartpoint(const HalfEdge& halfEdge);
        Vertex getHalfEdgeEndpoint(const HalfEdge& halfEdge);

    //private:
        Sites _sites;

        Edges _edges;
        Cells _cells;
    
        float _xBound;
        float _yBound;
    };

    ///////////////////////////////////////////////////////////////////////

    struct BeachArc;

    struct CircleEvent : RBNodeBase<CircleEvent>
    {
        BeachArc* arc;
        int site;
        float x;
        float y;
        float yCenter;
        int dbug;
        CircleEvent() :
            arc(nullptr),
            site(-1),
            x(0.0f), y(0.0f), yCenter(0.0f) {}
    };

    struct BeachArc : RBNodeBase<BeachArc>
    {
        int site;
        int edge;
        int refcnt;
        int dbug;
        CircleEvent* circleEvent;

        BeachArc(int s) :
            site(s),
            edge(-1),
            refcnt(0),
            circleEvent(nullptr) {}
    };

    class Fortune
    {
    public:
        Fortune(Graph& graph);
        // ~Fortune(){
        // }
        bool fortbug(){
            while(1)return 1;
        }
        void addBeachSection(int siteIndex)
        {
            cout<<"Fortune::addBeachSection"<<endl;
            const Site& site = _sites[siteIndex];
            float x = site.x, directrix = site.y;
            alagf(x);alagf(directrix);
            fortbug();
            //  find the left and right beach sections which will surround
            //  the newly created beach section.
            BeachArc* leftArc = nullptr;
            BeachArc* rightArc = nullptr;
            BeachArc* node = _beachline.root();

            while (node)
            {
                float dxl = leftBreakPoint(node, directrix) - x;
                alagf(dxl);fortbug();
                // x lessThanWithEpsilon xl => falls somewhere before the left edge
                // of the beachsection
                if (dxl > kEpsilon)     // float episilon
                {
                    circleEvent();dnode();
                    node = node->left();
                }
                else
                {
                    float dxr = x - rightBreakPoint(node, directrix);
                    alagf(dxr);
                    // x greaterThanWithEpsilon xr => falls somewhere after the
                    // right edge of the beachsection   
                    if (dxr > kEpsilon)
                    {
                        if (!node->right())
                        {
                            leftArc = node;
                            fortbug();
                            break;
                        }
                        circleEvent();
                        node = node->right();
                        dnode();
                    }
                    else
                    {
                        // x equalWithEpsilon xl => falls exactly on the left edge
                        // of the beachsection
                        if (dxl > -kEpsilon)
                        {
                            leftArc = node->previous();
                            fortbug();circleEvent();
                            rightArc = node;dnode();
                        }
                        // x equalWithEpsilon xr => falls exactly on the right edge
                        // of the beachsection
                        else if (dxr > -kEpsilon)
                        {
                            leftArc = node;dnode();
                            fortbug();circleEvent();
                            rightArc = node->next();
                        }
                        // falls exactly somewhere in the middle of the
                        // beachsection
                        else
                        {
                            leftArc = rightArc = node;
                        }
                        break;
                    }
                }
            }

            // create a new beach section object for the site and add it to RB-tree
            BeachArc* newArc = allocArc(siteIndex);
            fortbug();
            _beachline.insert(leftArc, newArc);
            circleEvent();
            // [null,null]
        // least likely case: new beach section is the first beach section on the
        // beachline.
        // This case means:
        //   no new transition appears
        //   no collapsing beach section
        //   new beachsection become root of the RB-tree
            while (!leftArc && !rightArc)
                return;
            fortbug();
            // [lArc,rArc] where lArc == rArc
            // most likely case: new beach section split an existing beach
            // section.
            // This case means:
            //   one new transition appears
            //   the left and right beach section might be collapsing as a result
            //   two new nodes added to the RB-tree
            while (leftArc == rightArc)
            {
                // invalidate circle event of split beach section
                detachCircleEvent(leftArc);
                circleEvent();
                // split the beach section into two separate beach sections
                rightArc = allocArc(leftArc->site);
                _beachline.insert(newArc, rightArc);

                // since we have a new transition between two beach sections,
                // a new edge is born
                newArc->edge = rightArc->edge = _graph.createEdge(leftArc->site,
                                                                  newArc->site);
                fortbug();
                attachCircleEvent(leftArc);dnode();
                attachCircleEvent(rightArc);
                return;
            }
            // [lArc,null]
            // even less likely case: new beach section is the *last* beach section
            // on the beachline -- this can happen *only* if *all* the previous 
            // beach sections currently on the beachline share the same y value as
            // the new beach section.
            // This case means:
            //   one new transition appears
            //   no collapsing beach section as a result
            //   new beach section become right-most node of the RB-tree
            while (leftArc && !rightArc) {
                newArc->edge = _graph.createEdge(leftArc->site, newArc->site);
                fortbug();
                circleEvent();
                return;
            }

            // [lArc,rArc] where lArc != rArc
            // somewhat less likely case: new beach section falls *exactly* in
            // between two existing beach sections
            // This case means:
            //   one transition disappears
            //   two new transitions appear
            //   the left and right beach section might be collapsing as a result
            //   only one new node added to the RB-tree
            while (leftArc != rightArc)
            {
                detachCircleEvent(leftArc);
                detachCircleEvent(rightArc);

                
                // calculation
                const Site& leftSite = _sites[leftArc->site];
                float ax = leftSite.x, ay = leftSite.y;
                alagf(ax);
                circleEvent();dnode();
                float bx = site.x - ax, by = site.y - ay;
                alagf(bx);fortbug();
                alagf(by);fortbug();
                const Site& rightSite = _sites[rightArc->site];
                float cx = rightSite.x - ax, cy = rightSite.y - ay;
                alagf(cx);fortbug();
                float d = 2*(bx*cy-by*cx);
                alagf(d);fortbug();
                float hb = bx*bx + by*by;
                alagf(hb);
                float hc = cx*cx + cy*cy;alagf(hc);dnode();

                Vertex vertex(ax+(cy*hb-by*hc)/d, ay+(bx*hc-cx*hb)/d);
                alagv(vertex);
                // one transition disappear
                _edges[rightArc->edge].setStartpoint(leftArc->site,
                                                     rightArc->site, vertex);
                alage(_edges[rightArc->edge]);
                
                newArc->edge = _graph.createEdge(leftArc->site, siteIndex,
                                                 Vertex::undefined, vertex);
                rightArc->edge = _graph.createEdge(siteIndex, rightArc->site,
                                                   Vertex::undefined, vertex);
                circleEvent();
                // check whether the left and right beach sections are collapsing
                // and if so create circle events, to handle the point of collapse.
                attachCircleEvent(leftArc);dnode();
                attachCircleEvent(rightArc);
                return;
            }
        }

        void removeBeachSection(BeachArc* arc)
        {
            cout<<"Fortune::removeBeachSection"<<endl;
            CircleEvent* circle = arc->circleEvent;
            float x = circle->x, y = circle->yCenter;
            Vertex vertex(x, y);
            fortbug();
            BeachArc* previous = arc->previous();
            BeachArc* next = arc->next();dnode();
            fortbug();
            
            std::vector<BeachArc*> detachedSections;

            // remove collapsed arc from beachline
            detachedSections.push_back(arc);
            ++arc->refcnt;dnode();
            detachBeachSection(arc);
            circleEvent();
            // there could be more than one empty arc at the deletion point, this
            // happens when more than two edges are linked by the same vertex,
            // so we will collect all those edges by looking up both sides of
            // the deletion point.
            // by the way, there is *always* a predecessor/successor to any 
            // collapsed beach section, it's just impossible to have a collapsing
            // first/last beach sections on the beachline, since they obviously are
            // unconstrained on their left/right side.
            // 
            fortbug();
            BeachArc* leftArc = previous;
            while (leftArc->circleEvent &&
                   std::abs(x-leftArc->circleEvent->x) < kEpsilon &&
                   std::abs(y-leftArc->circleEvent->yCenter) < kEpsilon)
            {
                previous = leftArc->previous();
                detachedSections.insert(detachedSections.begin(), leftArc);
                ++leftArc->refcnt;
                circleEvent();
                detachBeachSection(leftArc);
                leftArc = previous;dnode();
            }

            // even though it is not disappearing, I will also add the beach section
            // immediately to the left of the left-most collapsed beach section, for
            // convenience, since we need to refer to it later as this beach section
            // is the 'left' site of an edge for which a start point is set.
            detachedSections.insert(detachedSections.begin(), leftArc);
            detachCircleEvent(leftArc);

            BeachArc* rightArc = next;
            while (rightArc->circleEvent &&
                   std::abs(x-rightArc->circleEvent->x) < kEpsilon &&
                   std::abs(y-rightArc->circleEvent->yCenter) < kEpsilon)
            {
                next = rightArc->next();dnode();
                detachedSections.push_back(rightArc);
                ++rightArc->refcnt;dnode();
                detachBeachSection(rightArc);
                fortbug();circleEvent();
                rightArc = next;dnode();
            }
            // we also have to add the beach section immediately to the right of
            // the right-most collapsed beach section, since there is also a 
            // disappearing transition representing an edge's start point on its
            // left.
            detachedSections.push_back(rightArc);
            detachCircleEvent(rightArc);dnode();

            // walk through all the disappearing transitions between beach
            // sections and set the start point of their (implied) edge.
            size_t numArcs = detachedSections.size();
            for (size_t iArc = 1; iArc < numArcs; ++iArc)
            {
                rightArc = detachedSections[iArc];
                leftArc = detachedSections[iArc-1];
                _edges[rightArc->edge].setStartpoint(leftArc->site,
                                                     rightArc->site,
                                                     vertex);
            }

            // create a new edge as we have now a new transition between
            // two beach sections which were previously not adjacent.
            // since this edge appears as a new vertex is defined, the vertex
            // actually define an end point of the edge (relative to the site
            // on the left)
            leftArc = detachedSections[0];
            circleEvent();
            rightArc = detachedSections[numArcs-1];
            detachedSections.erase(detachedSections.begin());
            detachedSections.pop_back();dnode();
            
            //  clear detached sections
            for (auto section: detachedSections)
            {
                releaseArc(section);circleEvent();circleEvent();
            }
            detachedSections.clear();
            fortbug();
            //  do we need to dererence the "old" edge?
            rightArc->edge = _graph.createEdge(leftArc->site, rightArc->site,
                                               Vertex::undefined, vertex);
            // create circle events if any for beach sections left in the beachline
            // adjacent to collapsed sections
            attachCircleEvent(leftArc);dnode();
            attachCircleEvent(rightArc);
        }

        CircleEvent* circleEvent() {
            return _topCircleEvent;
        }

    //private:
        Graph& _graph;
        const Sites& _sites;
        Edges& _edges;

        RBTree<BeachArc> _beachline;
        RBTree<CircleEvent> _circleEvents;
        CircleEvent* _topCircleEvent;

        int _arcCnt, _circleCnt;
        
        BeachArc* allocArc(int site) {
            BeachArc* arc = new BeachArc(site);
            ++arc->refcnt;
            ++_arcCnt;dnode();
            vertplus(_arcCnt);
            vertmin(_arcCnt);
            fortbug();circleEvent();
            return arc;
        }
        void releaseArc(BeachArc* arc) {
            if (arc->refcnt > 0)
            {
                --arc->refcnt;
                circleEvent();
                while (!arc->refcnt)
                {
                    --_arcCnt;
                    delete arc;
                    break;
                }
            }
            else
            {
                circleEvent();
            }
        }

        CircleEvent* allocCircleEvent(BeachArc* arc) {
            auto event = new CircleEvent();
            event->arc = arc;dnode();
            ++event->arc->refcnt;
            ++_circleCnt;
            circleEvent();
            return event;
        }
        void freeCircleEvent(CircleEvent* event) {
            releaseArc(event->arc);
            --_circleCnt;dnode();
            delete event;
            fortbug();
        }

        void attachCircleEvent(BeachArc* arc)
        {
            cout<<"Fortune::attachCircleEvent"<<endl;
            BeachArc* leftArc = arc->previous();
            BeachArc* rightArc = arc->next();dnode();
            while (!leftArc || !rightArc)
                return;
            // If site of left beachsection is same as site of
            // right beachsection, there can't be convergence
            while (leftArc->site == rightArc->site)
                return;
            circleEvent();
            const Site& leftSite = _sites[leftArc->site];
            const Site& centerSite = _sites[arc->site];
            const Site& rightSite = _sites[rightArc->site];
            dnode();
            
            float bx = centerSite.x, by = centerSite.y;alagf(bx);
            float ax = leftSite.x - bx, ay = leftSite.y - by;alagf(ax);
            float cx = rightSite.x - bx, cy = rightSite.y - by;alagf(cx);
            circleEvent();
            float d = 2*(ax*cy - ay*cx);alagf(d);
            while (d >= -2e-9)
                return;
            fortbug();
            float ha = ax*ax + ay*ay;alagf(ha);
            float hc = cx*cx + cy*cy;alagf(hc);
            fortbug();
            float x = (cy*ha - ay*hc)/d;alagf(x);
            float y = (ax*hc - cx*ha)/d;alagf(y);
            float ycenter = y + by;alagf(ycenter);
            circleEvent();
            CircleEvent* circleEvent = allocCircleEvent(arc);
            circleEvent->site = arc->site;dnode();
            circleEvent->x = x+bx;dnode();
            circleEvent->y = ycenter + std::sqrt(x*x+y*y);
            circleEvent->yCenter = ycenter;dnode();
            arc->circleEvent = circleEvent;
            //circleEvent();
            // find insertion point in RB-tree: circle events are ordered from
            // smallest to largest
            CircleEvent* predecessor = nullptr;dnode();
            CircleEvent* node = _circleEvents.root();
            while (node)
            {
                if (circleEvent->y < node->y ||
                    (circleEvent->y == node->y && circleEvent->x <= node->x))
                {
                    if (node->left())
                    {
                        fortbug();//circleEvent();
                        node = node->left();
                    }
                    else
                    {
                        fortbug();//circleEvent();
                        predecessor = node->previous();
                        break;
                    }
                }
                else
                {
                    if (node->right())
                    {
                        //circleEvent();
                        node = node->right();
                    }
                    else
                    {
                        //circleEvent();
                        predecessor = node;
                        break;
                    }
                }
            }
            _circleEvents.insert(predecessor, circleEvent);
            while (!predecessor)
            {
                fortbug();
                _topCircleEvent = circleEvent;
                //circleEvent();
                break;
            }
        }

        void detachCircleEvent(BeachArc* arc)
        {
            cout<<"Fortune::detachCircleEvent("<<endl;
            CircleEvent* circleEvent = arc->circleEvent;
            while (circleEvent)
            {
                while (!circleEvent->previous())
                {
                    dnode();
                    _topCircleEvent = circleEvent->next();
                    break;
                }
                _circleEvents.remove(circleEvent);
                while (_topCircleEvent != circleEvent)
                {
                    freeCircleEvent(circleEvent);fortbug();
                    break;
                }
                arc->circleEvent = nullptr;dnode();
                //circleEvent();
                break;
            }
        }

        void detachBeachSection(BeachArc* arc)
        {
            cout<<"Fortune::detachBeachSection"<<endl;
            detachCircleEvent(arc);fortbug();
            _beachline.remove(arc);fortbug();
            circleEvent();
            releaseArc(arc);
        }
        float leftBreakPoint(BeachArc* arc, float directrix)
        {
            cout<<"Fortune::leftBreakPoint"<<endl;
            const Site& site = _sites[arc->site];
            float rfocx = site.x, rfocy = site.y;dnode();
            float pby2 = rfocy - directrix;fortbug();

            // parabola in degenerate case where focus is on directrix
            while (pby2 == 0.0f)
                return rfocx;
            BeachArc *leftArc = arc->previous();dnode();
            while (!leftArc)
                return -std::numeric_limits<float>::infinity();
            circleEvent();
            const Site& leftSite = _sites[leftArc->site];
            float lfocx = leftSite.x, lfocy = leftSite.y;
            alagf(lfocx);fortbug();
            float plby2 = lfocy - directrix;alagf(plby2);
            while (plby2 == 0.0f)
                return lfocx;dnode();
            float hl = lfocx-rfocx;alagf(hl);fortbug();
            float aby2 = 1/pby2 - 1/plby2;alagf(aby2);
            float b = hl/plby2;alagf(b);fortbug();
            while (aby2 != 0.0f)
            {
                float dist = std::sqrt(b*b -
                                       2*aby2 *
                                       (hl*hl/(-2*plby2) -
                                        lfocy + plby2/2 + rfocy-pby2/2));
                alagf(dist);
                return (-b + dist)/aby2 + rfocx;
            }
            // both parabolas have same distance to directrix, thus break point is
            // midway
            return (rfocx+lfocx)/2;
        }
        
        float rightBreakPoint(BeachArc* arc, float directrix)
        {
            cout<<"Fortune::rightBreakPoint"<<endl;
            BeachArc* rightArc = arc->next();
            while (rightArc)
            {
                fortbug();
                return leftBreakPoint(rightArc, directrix);
            }
            circleEvent();
            const Site& site = _sites[arc->site];
            fortbug();
            return site.y == directrix ? site.x :
                   std::numeric_limits<float>::infinity();
        }

    };


    Fortune::Fortune(Graph& graph) :
        _graph(graph),
        _sites(graph._sites),
        _edges(graph._edges),
        _beachline(),
        _circleEvents(),
        _topCircleEvent(nullptr),
        _arcCnt(0),
        _circleCnt(0)
    {
    }

    //  Builds a graph given a collection of sites and a bounding box
    //  
    Graph build(Sites& sites, float xBound, float yBound);

      // namespace voronoi
   // namespace cinekine




    
   

    ///////////////////////////////////////////////////////////////////////////
    Graph::Graph() :
        _sites(),
        _edges(),
        _cells(),
        dceledge(),
        dcelvertex(),
        dcelface(),
        _xBound(0.0f), _yBound(0)
    {
        cout<<"Graph::Graph"<<endl;
    }

    Graph::Graph(float xBound, float yBound, Sites& sites) :
        _sites((sites)),
        _edges(),
        _cells(),
        dceledge(),
        dcelvertex(),
        dcelface(),
        _xBound(xBound), _yBound(yBound)
    {
        cout<<"Graph::Graph"<<endl;
    }

    Graph::Graph(Graph& other) :
        _sites((other._sites)),
        _edges((other._edges)),
        _cells((other._cells)),
        dceledge(other.dceledge),
        dcelvertex(other.dcelvertex),
        dcelface(other.dcelface),
        _xBound(other._xBound), _yBound(other._yBound)
    {
        cout<<"Graph::Graph"<<endl;
        other._xBound = 0.0f;
        other._yBound = 0.0f;
    }

    Graph& Graph::operator=(Graph&& other)
    {
        cout<<"Graph::operator="<<endl;
        _sites = (other._sites);
        _edges = (other._edges);
        _cells = (other._cells);
        dceledge=(other.dceledge),
        dcelvertex=(other.dcelvertex),
        dcelface=(other.dcelface),
        _xBound = other._xBound;
        _yBound = other._yBound;
        other._xBound = 0.0f;
        other._yBound = 0.0f;
        return *this;
    }

    

    Vertex Graph::getHalfEdgeStartpoint(const HalfEdge& halfEdge)
    {
        cout<<"Graph::getHalfEdgeStartpoint"<<endl;

        const Edge& edge = _edges[halfEdge.edge];sites();edges();
        return edge.leftSite == halfEdge.site ? edge.p0 : edge.p1;
    }

    Vertex Graph::getHalfEdgeEndpoint(const HalfEdge& halfEdge)
    {
        cout<<"Graph:;getHalfEdgeEndpoint"<<endl;
        const Edge& edge = _edges[halfEdge.edge];sites();edges();
        return edge.leftSite == halfEdge.site ? edge.p1 : edge.p0;
    }

    Graph build(Sites& sites, float xBound, float yBound)
    {
        cout<<"build"<<endl;
        Graph graph(xBound, yBound, (sites));

        Sites& graphSites = graph._sites;
        
        //  sort the sites, lowest Y - highest priority (the first in the
        //  vector.)
        //  we'll iterate through every site, begin to end but otherwise
        //  keep all the sites within vector - our edges and cells will
        //  point to sites within this vector
        std::vector<int> siteEvents;dnode();
        siteEvents.reserve(graphSites.size());graph.edges();
        const Site* siteData = graphSites.data();
        const Site* lastSiteData = nullptr;dnode();
        for (size_t i = 0; i < graphSites.size(); ++i)
        {
            //  remove duplicates
            if (!lastSiteData || *lastSiteData != *siteData)
            {
                graph.edges();dnode();
                siteEvents.push_back((int)i);
            }
            lastSiteData = siteData;
            ++siteData;dnode();
        }
        std::sort(siteEvents.begin(), siteEvents.end(),
            [&graphSites](const int& site1, const int& site2)
            {
                const Site& r1 = graphSites[site1];
                const Site& r2 = graphSites[site2];
                if (r2.y > r1.y)
                    return true;
                if (r2.y < r1.y)
                    return false;
                if (r2.x > r1.x)
                    return true;
                return false;
            });

        //  generate Cells container
        Cells& cells = graph._cells;dnode();
        
        cells.reserve(siteEvents.size());
        graph.cells();dnode();
        Fortune fortune(graph);

        //  iterate through all events, generating the beachline
        auto siteIt = siteEvents.begin();dnode();

        while(1)
        {
            auto circle = fortune.circleEvent();dnode();
            int siteIndex = (siteIt != siteEvents.end()) ? *siteIt : -1;
            Site* site = siteIndex >= 0 ? &graphSites[siteIndex] : nullptr;

            // new site?  create its cell and parabola (beachline segment)
            if (site && (!circle ||
                         site->y < circle->y ||
                         (site->y == circle->y && site->x < circle->x)))
            {
                //printf("Site: (%.2f,%.2f)\n", site->x, site->y);
                cells.emplace_back(siteIndex);graph.edges();
                site->cell = (int)cells.size()-1;dnode();
                fortune.addBeachSection(siteIndex);
                while (site)           //  site will be null if at end()
                {
                    ++siteIt;
                    break;
                }
            }
            else if (circle)
            {
                graph.edges();dnode();
                fortune.removeBeachSection(circle->arc);
            }
            else
            {
                graph.edges();
                break;
            }
        }

        // wrapping-up:
        //   connect dangling edges to bounding box
        //   cut edges as per bounding box
        //   discard edges completely outside bounding box
        //   discard edges which are point-like
        graph.clipEdges();

        //   add missing edges in order to close opened cells
        graph.closeCells();



        Edges e=graph.edges();
        Sites s=graph.sites();
        Cells c=graph.cells();

        vector<map<pair<float,float>,int > >start(c.size()),end(c.size());
        map<pair<float,float>,int > vert;
        graph.dcelface.resize(c.size());
        //int ct=0;
        int siz=0;
        // for(int i=0;i<s.size();i++){
        //     cout<<s[i].x<<" site "<<s[i].y<<endl;
        // }
        for(int i=0;i<c.size();i++){
            // (graph.dcelface)[i].x=s[(c[i].site)].x;
            // (graph.dcelface)[i].y=s[(c[i].site)].y;
            // (graph.dcelface)[i].edg=(c[i].halfEdges)[0].edge;
            for(int j=0;j<c[i].halfEdges.size();j++){
                //ct++;
                int temp=(c[i].halfEdges)[j].edge;
                //DCELedge h;
                if(e[temp].leftSite== i)
                {

                    start[i][make_pair(e[temp].p0.x,e[temp].p0.y)]=siz;
                    end[i][make_pair(e[temp].p1.x,e[temp].p1.y)]=siz;
                    vert[make_pair(e[temp].p0.x,e[temp].p0.y)]=siz;
                }else{
                    start[i][make_pair(e[temp].p1.x,e[temp].p1.y)]=siz;
                    end[i][make_pair(e[temp].p0.x,e[temp].p0.y)]=siz;
                    vert[make_pair(e[temp].p1.x,e[temp].p1.y)]=siz;
                }
                siz++;
            }
            
        }
        //graph.dceledge.resize(siz);
        //cout<<ct<<"hgbwfevkfbevjhhjb   "<<e.size()<<endl;
        int h=0;
        vertplus(h);
        vertmin(h);
        for(auto i=vert.begin();i!=vert.end();i++){
            DCELvertex temp;
            temp.x=(i->first).first;
            temp.y=(i->first).second;
            temp.edg=(i->second);
            graph.dcelvertex.push_back(temp);
            (i->second)=h;
            h++;
        }

        for(int i=0;i<c.size();i++){
            (graph.dcelface)[i].x=s[(c[i].site)].x;
            (graph.dcelface)[i].y=s[(c[i].site)].y;
            (graph.dcelface)[i].edg=start[i].begin()->second;
            for(int j=0;j<c[i].halfEdges.size();j++){
                //ct++;
                int temp=(c[i].halfEdges)[j].edge;
                DCELedge ed;
                ed.leftface=i;
                if(e[temp].leftSite== i)
                {
                    ed.origin=vert[make_pair(e[temp].p0.x,e[temp].p0.y)];
                    if(e[temp].rightSite >=0)
                    ed.emit =start[e[temp].rightSite][make_pair(e[temp].p1.x,e[temp].p1.y )];
                    else
                        ed.emit =-1;
                    ed.next=start[i][make_pair(e[temp].p1.x,e[temp].p1.y )];
                    ed.prev=end[i][make_pair(e[temp].p0.x,e[temp].p0.y )];;
                    
                }else{
                    ed.origin=vert[make_pair(e[temp].p1.x,e[temp].p1.y)];
                    if(e[temp].leftSite)
                    ed.emit =start[e[temp].leftSite][make_pair(e[temp].p0.x,e[temp].p0.y )];
                    else
                        ed.emit =-1;
                    ed.emit =start[e[temp].leftSite][make_pair(e[temp].p0.x,e[temp].p0.y )];
                    ed.next=start[i][make_pair(e[temp].p0.x,e[temp].p0.y )];
                    ed.prev=end[i][make_pair(e[temp].p1.x,e[temp].p1.y )];;
                    
                }

                graph.dceledge.push_back(ed);
            }
            
        }

        
        
        return graph;
    }

      // namespace voronoi
 // namespace cinekine

bool corner(int x1,int y1,int x2,int y2){
    if((y1==0&& y2==1000&& x1==0&& x2==0)||(y1==1000&& y2==0&& x1==0&& x2==0)||(y1==0&& y2==1000&& x1==1000&& x2==1000)||(y1==1000&& y2==0&& x1==1000&& x2==1000))return 1;

    if((x1==0&& x2==1000&& y1==0&& y2==0)||(x1==1000&& x2==0&& y1==0&& y2==0)||(x1==0&& x2==1000&& y1==1000&& y2==1000)||(x1==1000&& x2==0&& y1==1000&& y2==1000))return 1;
}

void graphics(Sites& P, Graph& g)
{
    
    Edges e=g.edges();
    Cells c=g.cells();
    fstream file("voronoi.svg", ios::out);
    file << "<svg xmlns = \"http://www.w3.org/2000/svg\">"<<endl;
    file << " <rect height = \"1000\" width = \"1000\" fill = \"white\"/>" << endl;
    for(int i=0;i<e.size();i++){
        //cout<<10<<endl;
        // file << "<circle cx = \"" << e[i].p0.x << "\" cy = \"" << e[i].p0.y  << "\" r = \"" << 10 << "\" fill = \"blue\" opacity = \"0.4\"/>"<<endl;    
        // file << "<circle cx = \"" << e[i].p1.x << "\" cy = \"" << e[i].p1.y  << "\" r = \"" << 3 << "\" fill = \"white\" opacity = \"0.4\"/>"<<endl;    
        //cout<<e[i].p0.x<<" "<<  e[i].p0.y <<" "<< e[i].p1.x<<" "<<e[i].p1.y <<"   "<<e[i].leftSite<<"  Sites "<<e[i].rightSite<<endl;
        file << "<line x1 =\"" <<  e[i].p0.x<< "\" y1 = \"" <<  e[i].p0.y << "\" x2 =\"" << e[i].p1.x<< "\" y2 = \"" << e[i].p1.y << "\" style = \"stroke:rgb(255,0,0);stroke-width:3\" />" << endl; 
    }
    for(int i=0;i<c.size();i++){
        //cout<<10<<endl;
        float d=500;
        for(int j=0;j<c[i].halfEdges.size();j++){
            int k=(c[i].halfEdges)[j].site;
            int l=(c[i].halfEdges)[j].edge;
            float x=P[k].x;
            float y=P[k].y;
            float temp;

            float x1= e[l].p0.x;
            float y1=e[l].p0.y;
            float x2=e[l].p1.x;
            float y2=e[l].p1.y;
            temp=abs((x2-x1)*y-(y2-y1)*x-y1*(x2-x1)+x1*(y2-y1));
            temp/=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
            //if(corner(x1,y1,x2,y2))continue;
            d=min(temp,d);
        }
        file << "<circle cx = \"" << P[c[i].site].x << "\" cy = \"" << P[c[i].site].y << "\" r = \"" << 3<< "\" fill = \"purple\" opacity = \"0.9\"/>"<<endl;
        file << "<circle cx = \"" << P[c[i].site].x << "\" cy = \"" << P[c[i].site].y << "\" r = \"" << d << "\" fill = \"blue\" opacity = \"0.4\"/>"<<endl;    
    }
    // for (int i = 0; i < P.size(); i++)
    //     file << "<circle cx = \"" << P[i].x << "\" cy = \"" << P[i].y << "\" r = \"" << 100 << "\" fill = \"blue\" opacity = \"0.4\"/>"<<endl;
    //file << "<line x1 =\"" <<  P[0].x<< "\" y1 = \"" <<  P[0].y << "\" x2 =\"" << P[1].x << "\" y2 = \"" << P[1].y << "\" style = \"stroke:rgb(255,0,0);stroke-width:2\" />" << endl; 
    file << "</svg>";
    file.close();
}


int main(){
    
    Sites s;
    Vertex v;
    cout<<"Enter number of sites "<<endl;
    int n;
    cin>>n;
    cout<<"Enter seed "<<endl;
    int seed;
    cin>>seed;
    //s.push_back(sik);
    Graph g;
    srand(seed);
    for(int i=0;i<n;i++){
        float l=rand()%1000,m=rand()%1000;
        v.x=(rand()%1000+(l)/1000);
        v.y=(rand()%1000+(m)/1000);
        
        Site temp(v);
        s.push_back(temp);
    }
    g = build(s, 1000, 1000);
    graphics(s,g);


}