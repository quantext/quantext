/**********************************************************************
	wordtree.js
	Utilities for turning a list of sentences into a graphical display

Copyright (c) 2011 Aditi Muralidharan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
**********************************************************************/

/** makes a tree out of either a left or a right context
	@param context : a list of strings (sentences);
 **/
function makeTree(context, level, detail, parent){
	var tree = {};
	var first, sentence, max, collapsed, subtree, name ;
	max = 0;
	var keys = [];
	for(var i = 0; i < context.length; i++){
		sentence = context[i];
		if(sentence.length > 0){
			first = sentence[0];
			if(!first){
				sentence = sentence.slice(1);
				first = sentence[0]
			}
			if(first){
				first = first.toLowerCase();
				if(tree[first]== undefined){
					tree[first] = {parent:parent,name:first, after:[],level:level, count:1, children:[]};
				}
				tree[first].after.push(sentence.slice(1));
				tree[first].count += 1;
			} 
		}
	}
	for(var name in tree){
		if(tree[name].count > max){
			max = tree[name].count;
		}
		keys.push(name);
	}
	for(var i = 0 ; i < keys.length; i++){
		name = keys[i];
		if(tree[name].count > (max*(100-detail)/100)){
			if(tree[name].after.length > 1){
				subtree = makeTree(tree[name].after, level+1, detail,tree[name]["name"]);
				collapsed = collapse(subtree, name);
				tmp = tree[name];
				delete tree[name];
				tree[collapsed.name] = {};
				tree[collapsed.name]["parent"] = tmp["parent"];
				tree[collapsed.name]["after"] = tmp["after"];
				tree[collapsed.name]["count"] = tmp["count"];
				tree[collapsed.name]["level"] = tmp["level"];
				tree[collapsed.name]["children"] = collapsed.children;
				tree[collapsed.name]["name"] = collapsed.name;
			}
		}else{
			delete tree[name];
		}
	}
	return sort(tree);
}

function compareSubTrees(t1, t2){
	return t2.count - t1.count;
}

function sort(tree){
	var tmp = [];
	for(name in tree){
		tmp.push(tree[name]);
	}
	tmp.sort(compareSubTrees);
	return tmp
}

function collapse(tree, name){
	if(tree.length == 1){
		for(k in tree){
			return collapse(tree[k].children, name+" "+tree[k].name);
		}
	}else{
		return {children:tree, "name":name}
	}
}

function size(count, level){
	if(count == "end"){
		return 10;
	}else{
		return Math.min(30, Math.max(12, (12+count)/(Math.log(level+1))));
	}
}
/** 
@param context: the name of the root node 
**/


function makeContext(data,type, which){
		return data[type][which]
}

function makeWordTree(sentences, context, detail, div, number){
	var children = makeTree(sentences, 1, detail, context);
	var tree = {"name":context, "children":children,parent:null,"data":{"count":sentences.length}};
	drawTree(tree,div,number);
	//return displayTree(context, tree, container, width, height, direction, paper)
}

function drawTree(treeData,div,number){
	var max = (treeData.data.count) ? treeData.data.count : 10;

	var margin = {top: 20, right: 90, bottom: 30, left: 90},
		width = 960 - margin.left - margin.right,
		height = 500 - margin.top - margin.bottom;

	var shiftKey;

// append the svg object to the body of the page
// appends a 'group' element to 'svg'
// moves the 'group' element to the top left margin
	var svg = d3.select(div).html("").append("svg")
		.attr("width", "100%")
		.attr("height", height + margin.top + margin.bottom)
		.call(d3.zoom().on("zoom", function () {
			svg.attr("transform", d3.event.transform);
		}))
		.append("g")
		.attr("transform", "translate("
			+ margin.left + "," + margin.top + ")");

	width = d3.select("svg").node().getBBox().width;

	var i = 0,
		duration = 750,
		root;

// declares a tree layout and assigns the size
	var treemap = d3.tree().size([height, width]);

// Assigns parent, children, height, depth
	root = d3.hierarchy(treeData, function(d) { return d.children; });
	root.x0 = height / 2;
	root.y0 = 0;

// Collapse after the second level
	root.children.forEach(collapse);

	update(root);

	rect = svg.append('rect')
		.attr('pointer-events', 'all')
		.attr('width', width)
		.attr('height', height)
		.style('fill', 'none');

// Collapse the node and all it's children
	function collapse(d) {
		if(d.children) {
			d._children = d.children
			d._children.forEach(collapse)
			d.children = null
		}
	}

	function update(source) {

		// Assigns the x and y position for the nodes
		var treeData = treemap(root);

		// Compute the new tree layout.
		var nodes = treeData.descendants(),
			links = treeData.descendants().slice(1);

		// Normalize for fixed-depth.
		nodes.forEach(function(d){ d.y = d.depth * 180});

		// ****************** Nodes section ***************************

		// Update the nodes...
		var node = svg.selectAll('g.node')
			.data(nodes, function(d) {return d.id || (d.id = ++i); });

		// Enter any new modes at the parent's previous position.
		var nodeEnter = node.enter().append('g')
			.attr('class', 'node node-'+number)
			.attr("transform", function(d) {
				//var width = chainWidths(d);
				return "translate(" + source.y0 + "," + source.x0 + ")";
			})
			.on('click', click);

		// Add Circle for the nodes
	/*	nodeEnter.append('circle')
			.attr('r', 1e-6)
			.attr("class", function(d) {
				return d._children ? "node fill-"+number : "node not-filled";
			})*/

		function chainWidths(d) {
			if (d.parent)
				return d.parent.width + chainWidths(d.parent);
			else
				return 0;
		}

		// Add labels for the nodes
		var text = nodeEnter.append('text')
			.attr("class", function(d){
				return d.children || d._children ? "fill-"+number : "not-filled";
			})
			.attr("dy", ".35em")
			.attr("text-anchor", function(d) {
				//return d.children || d._children ? "end" : "start";
				return "start";
			})
			.text(function(d) { return d.data.name; })
			.style("font-size", function(d){
				var size = 10;
				if(d.data.count)
					size = (d.data.count - 1) * (30 / (max - 1)) + 10;
				else
					size = (d.data.data.count - 1) * (30 / (max - 1)) + 10;
				return size + "px";
			}).each(function(d){
				d.width = this.getBBox().width;
			});

		nodeEnter.attr("transform", function(d) {
			var width = chainWidths(d);
			return "translate(" + (width) + "," + source.x0 + ")";
		});

		// UPDATE
		var nodeUpdate = nodeEnter.merge(node);

		nodeUpdate.attr("transform", function(d){
			var width = chainWidths(d);
			return "translate(" + (source.y0+width) + "," + source.x0 + ")";
		});

		// Transition to the proper position for the node
		//NEED THIS ONE
		nodeUpdate
			.attr("transform", function(d) {
				var width = chainWidths(d);
				return "translate(" + (d.y + width) + "," + d.x + ")";
			});

		// Update the node attributes and style
		/*nodeUpdate.select('circle.node')
			.attr('r', 10)
			.attr("class", function(d) {
				return d._children ? "fill-"+number : "not-filled";
			})
			.attr('cursor', 'pointer');*/


		// Remove any exiting nodes
		var nodeExit = node.exit()
			.attr("transform", function(d) {
				var width = chainWidths(d);
				return "translate(" + (source.y+width) + "," + source.x + ")";
			})
			.remove();

		// On exit reduce the node circles size to 0
		nodeExit.select('circle')
			.attr('r', 1e-6);

		// On exit reduce the opacity of text labels
		nodeExit.select('text')
			.style('fill-opacity', 1e-6);

		// ****************** links section ***************************

		// Update the links...
		var link = svg.selectAll('path.link')
			.data(links, function(d) { return d.id; });

		// Enter any new links at the parent's previous position.
		var linkEnter = link.enter().insert('path', "g")
			.attr("class", "link")
			.attr('d', function(d){
				var o = {x: source.x0, y: source.y0}
				return diagonal(o, o)
			}).attr("transform", function(d){
				var width = chainWidths(d);
				return "translate("+width+",0)";
			});

		// UPDATE
		var linkUpdate = linkEnter.merge(link);

		// Transition back to the parent element position
		linkUpdate
			.attr('d', function(d){ return diagonal(d, d.parent) });

		// Remove any exiting links
		var linkExit = link.exit()
			.attr('d', function(d) {
				var o = {x: source.x, y: source.y}
				return diagonal(o, o)
			})
			.remove();

		// Store the old positions for transition.
		nodes.forEach(function(d){
			var width = chainWidths(d);
			d.x0 = d.x + width;
			d.y0 = d.y;
		});

		// Creates a curved (diagonal) path from parent to the child nodes
		function diagonal(s, d) {
			path = `M ${s.y} ${s.x}
            C ${(s.y + d.y) / 2} ${s.x},
              ${(s.y + d.y) / 2} ${d.x},
              ${d.y} ${d.x}`

			return path
		}

		// Toggle children on click.
		function click(d) {
			if (d.children) {
				d._children = d.children;
				d.children = null;
			} else {
				d.children = d._children;
				d._children = null;
			}

			update(d);
		}
	}
}