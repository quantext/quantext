let labeller = {
    select_div : target => {
        let labeller_div = target.matches(".labeller_div") ? target : target.closest(".labeller_div");
        let parent = labeller_div.closest(".label_panel");
        parent.querySelectorAll(".labeller_div.selected").forEach(div => {
            div.classList.remove("selected");
        });
        labeller_div.classList.add("selected");
        parent.querySelectorAll(".inner_text")[0].innerHTML = labeller_div.querySelectorAll(".text_div")[0].innerHTML;

        let input = parent.querySelectorAll(".category_input")[0];
        if(input){
            input.value = "";
            input.focus();
        }

        let categories = JSON.parse(labeller_div.dataset.categories);
        parent.querySelectorAll(".label_category").forEach(div => {
            div.classList.remove("selected");
            if (categories.includes(div.dataset.key))
                div.classList.add("selected");
        });
        parent.querySelectorAll(".all_responses_div")[0].scrollTop = labeller_div.offsetTop;
    },
    label_category : (target, cat_col) => {
        let container = target.closest(".parentContainer");
        let question_number = container.dataset.question;
        let parent = target.closest(".label_panel");
        let labeller_div = parent.querySelectorAll(".labeller_div.selected")[0];

        let clazz = "";
        target.classList.forEach(clazzes => {
            if(clazzes.indexOf("little_bar_") !== -1)
                clazz = clazzes;
        });

        let category = target.dataset.key;
        let categories = JSON.parse(labeller_div.dataset.categories);

        let total = parseInt(parent.querySelectorAll(".label_total")[0].innerHTML);
        let prev_selected = target.classList.contains("selected");
        let labels = labeller_div.querySelectorAll(".response_labels")[0];

        let master_labels = document.querySelectorAll(`.master_label_${category}`)[0] ? true : false;

        let used = null;
        if(master_labels)
            used = parseInt(document.querySelectorAll(`.master_label_${category}`)[0].querySelectorAll(".used")[0].innerHTML);

        if (prev_selected) {
            target.classList.remove("selected");
            categories = categories.filter(value => {
                return value !== category;
            });
            total--;            
            labels.querySelectorAll(`span.${clazz}`)[0].style.display = "none";
            if(master_labels)
                document.querySelectorAll(`.master_label_${category}`)[0].querySelectorAll(".used")[0].innerHTML = used-1;
        }
        else {
            target.classList.add("selected");
            categories.push(category);
            total++;
            labels.querySelectorAll(`span.${clazz}`)[0].style.display = "block";
            if(master_labels)
                document.querySelectorAll(`.master_label_${category}`)[0].querySelectorAll(".used")[0].innerHTML = used+1;
        }

        parent.querySelectorAll(".label_total")[0].innerHTML = total;

        parent.querySelectorAll(".label_bar").forEach(div => {
            let val = parseInt(div.dataset.value);
            if (div.dataset.key === category) {
                val += (!prev_selected) ? 1 : -1;
                div.dataset.value = val;
                div.setAttribute("title", `${category} : ${val}`);
                div.nextElementSibling.querySelectorAll(".label_bar_label_span")[0].innerHTML = val;
            }

            let newpercent = val / total * 100;
            div.style.width = `${newpercent}%`
        });

        let id = parent.querySelectorAll(".labeller_div.selected")[0].dataset.id;
        let data = {
            "selected": target.classList.contains("selected"),
            "category": category,
            "id": id,
            "analysis_id": _analysis_id,
            "file_id": _file_id,
            "question_number": question_number,
            "cat_col":cat_col
        };
        postData("/label_response", data).then(response => {  });
        
        labeller_div.classList.add("labelled")        
        labeller_div.dataset.categories = JSON.stringify(categories);

        let marked_up = parent.querySelectorAll(".labelled").length;
        parent.querySelectorAll(".marked_up_count")[0].innerHTML = marked_up;
        parent.querySelectorAll(".marked_up_percent")[0].innerHTML = Math.round((marked_up / parent.querySelectorAll(".labeller_div").length) * 100);
    },
    update_category : (target, destination, apply) => {
        let text = target.value;
        let question_number = target.dataset.question_number;
        let cat_col = document.getElementById("manual_label_sets") ? document.getElementById("manual_label_sets").value : "1";
        let parent = target.closest(".master_label");
        let key = parent.dataset.key;
        let data = {
            analysis_id:_analysis_id,
            file_id:_file_id,
            question_number:question_number,
            label:text,
            key:key,
            cat_col:cat_col
        };
        postData("/update_label", data).then(response => response.json()).then(result => {
            if(document.getElementById("set_histograms"))
                document.getElementById("set_histograms").innerHTML = result.histograms;
            let label_text = parent.querySelectorAll(".master_label_text")[0];
            label_text.innerHTML = text;
            label_text.style.display = "block";
            target.style.display = "none";            
            destination.querySelectorAll(`.category_label_${key}`)[0].innerHTML = text;
            
            let panel = destination.closest(".label_panel");

            let curr_label_bars = panel.querySelectorAll(".label_bars_div")[0];
            let expanded = false;
            curr_label_bars.querySelectorAll(".label_bars")[0].classList.forEach(clazz => {
                if(clazz.indexOf("expanded") !== -1)
                    expanded = true;
            });
            curr_label_bars.innerHTML = result.label_bars;

            if(expanded)
                curr_label_bars.querySelectorAll(".label_bars")[0].classList.add("expanded");
        });
    },
    new_category : (target, destination, apply) => {
        let text = target.value;
        let question_number = target.dataset.question_number;
        let cat_col = document.getElementById("manual_label_sets") ? document.getElementById("manual_label_sets").value : "1";
        let data = {
            analysis_id:_analysis_id,
            file_id:_file_id,
            question_number:question_number,
            label:text,
            cat_col:cat_col
        }
        postData("/new_label", data).then(response => response.json()).then(result => {
            if(document.getElementById("set_histograms"))
                document.getElementById("set_histograms").innerHTML = result.histograms;

            if(document.getElementById("custom_word_list"))
                document.getElementById("custom_word_list").insertAdjacentHTML("beforeend",result.master_label);
            
            target.value = "";
            
            let category = result.category            
            destination.insertAdjacentHTML("beforeend",category);

            let clazz = `label_bar_${(destination.querySelectorAll('.label_category').length-1)%15}`;
            let little_clazz = `little_bar_${destination.querySelectorAll('.label_category').length-1}`;

            let last_cat = destination.querySelectorAll(".label_category")[destination.querySelectorAll(".label_category").length-1];
            let parent = destination.closest(".label_panel");

            let curr_label_bars = parent.querySelectorAll(".label_bars_div")[0];
            let expanded = false;
            curr_label_bars.querySelectorAll(".label_bars")[0].classList.forEach(clazz => {
                if(clazz.indexOf("expanded") !== -1)
                    expanded = true;
            });
            curr_label_bars.innerHTML = result.label_bars;

            if(expanded)
                curr_label_bars.querySelectorAll(".label_bars")[0].classList.add("expanded");

            let label_span = document.createElement("span");
            label_span.classList.add(clazz,little_clazz);
            label_span.style.width = "5px";
            label_span.style.height = "16px";
            label_span.style.marginLeft = "1px";

            parent.querySelectorAll(".labeller_div").forEach(div => {
                let new_span = label_span.cloneNode(true);
                new_span.style.display = div.classList.contains(".selected") ? "block" : "none";                 
                div.querySelectorAll(".response_labels")[0].append(new_span);
            });
                            
            if(apply)
                last_cat.click();            
        });
    },
    toggle_label_control : target => {
        let question_number = target.dataset.question_number;
        let val = target.checked;
        let col = target.dataset.col;
        let data = {
            analysis_id: _analysis_id,
            file_id: _file_id,
            question_number: question_number,
            col:col
        };
        data['key'] = target.getAttribute("name");
        data['val'] = val;
        let container = event.target.closest(".label_panel");
        let parent = container.querySelectorAll(".parent_response_div")[0];
        if(target.matches(".hide_labels")){
            postData("/label_settings", data).then(response => response.text()).then(result => {
                if(target.checked) {
                    document.querySelectorAll(".labeller_div").forEach(d => {
                        d.classList.add("hide_labels_control")
                    });      
                    if(parent.querySelectorAll(".labeller_div:not(.labelled)").length > 0)        
                        parent.querySelectorAll(".labeller_div:not(.labelled)")[0].click();
                    else
                        parent.querySelectorAll(".labeller_div")[0].click();

                } else {
                    document.querySelectorAll(".labeller_div").forEach(d => {
                        d.classList.remove("hide_labels_control")
                    });
                }
            });
        }
    },
    trigger_bars : target => {
        let label_bars = target.matches(".label_bars") ? target : target.closest(".label_bars");
        if (label_bars.classList.contains("expanded"))
            label_bars.classList.remove("expanded");
        else
            label_bars.classList.add("expanded");
    },
    randomise_labels : target => {        
        let container = target.closest(".label_panel");
        let parent = container.querySelectorAll(".parent_response_div")[0];
        let responses = parent.querySelectorAll(".labeller_div");
        responses = Array.prototype.slice.call(responses, 0);
        responses = shuffleArray(responses);
        parent.innerHTML = "";
        responses.forEach(div => {
            parent.append(div);
        });
        if(container.querySelectorAll(".hide_labels")[0].checked)
            parent.querySelectorAll('.labeller_div:not(.labelled)')[0].click();
        else
            parent.querySelectorAll('.labeller_div')[0].click();        

        let sort_info = container.querySelectorAll(".sorting_info")[0];
        sort_info.style.display = "block";
        container.querySelectorAll(".sort_value")[0].innerHTML = "random";        
    }
}