#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:28:10 2017

@author: jenny
"""

#from summarise:

#
#         <h5>Question ID: {{question}}</h5>
#         <h5>Responses: {{sum_q(filename, question)[2]['count']}}</h5>
#         <h5>Response Length (characters):</h5>
#         <h5>Min = {{sum_q(filename, question)[2]['min']}}</h5>
#         <h5>Max = {{sum_q(filename, question)[2]['max']}}</h5>
#         <h5>Median = {{sum_q(filename, question)[2]['median']}}</h5>
#         <h5>Mean = {{sum_q(filename, question)[2]['mean']}}</h5>

def get_endpoint_args(request):
    "Given a Flask request, return combination of request & form args."
    endpoint_args = dict()
    if request.args:
        endpoint_args.update(request.args)
    if request.form:
        for k in request.form:
            endpoint_args.update([(k,request.form[k])])
    return endpoint_args
    
    @app.context_processor
def add_template_helpers():
 def _add_template_helpers(request):
     return get_endpoint_args(request)
 return dict(add_template_helpers = _add_template_helpers)
 
             <form class="form-inline" method="POST" action="{{ url_for('summarise', filename=filename) }}">
              <div class="form-group">
                <div class="input-group">
                    <span class="input-group-addon">Please select question to summarise</span>
                       <select class="form-control" name="selectq">
                        {%for q in filename | getqlist %}
                         <option value={{q}}>{{q}}</option>
                        {% endfor %}
                       </select>
                </div>
            <button type="submit" class="btn btn-primary">Summarise</button>
           </div>
         </form>