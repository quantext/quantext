document.addEventListener("click", event => {
    if (event.target)
        if (event.target.closest(".sidebar-div")) {
            let target = event.target.matches(".sidebar-div") ? event.target : event.target.closest(".sidebar-div");
            let panel = target.dataset.panel;
            let $panel = document.querySelectorAll(`.${panel}_panel`)[0];
            window.menuitem = target;
            if(target.matches('.clicked'))
            {
                target.classList.remove("clicked");
                $panel.classList.remove("shown");
            }
            else
            {
                if(document.querySelectorAll(".corpora_panel.shown").length > 0)
                    document.querySelectorAll(".corpora_panel.shown")[0].classList.remove("shown");
                if(document.querySelectorAll(".sidebar-div.clicked").length > 0) 
                    document.querySelectorAll(".sidebar-div.clicked")[0].classList.remove("clicked");
                target.classList.add("clicked");
                $panel.classList.add("shown");
            }
        } else if(event.target.closest(".close_corpora_panel")){
            let panel = event.target.closest(".corpora_panel");
            panel.classList.remove("shown");
            let menuitem = window.menuitem;
            menuitem.classList.remove("clicked");
        } else if(event.target.closest(".mobile_bar_div")){
            let target = event.target.matches(".mobile_bar_div") ? event.target : event.target.closest(".mobile_bar_div");
            let panel = target.dataset.panel;
            let $panel = document.querySelectorAll(`.${panel}_panel`)[0];
            window.menuitem = target;
            if(target.matches('.clicked'))
            {
                target.classList.remove("clicked");
                $panel.classList.remove("shown");
            }
            else
            {
                if(document.querySelectorAll(".corpora_panel.shown").length > 0)
                    document.querySelectorAll(".corpora_panel.shown")[0].classList.remove("shown");

                if(document.querySelectorAll(".mobile_bar_div.clicked").length > 0)
                    document.querySelectorAll(".mobile_bar_div.clicked")[0].classList.remove("clicked");
                target.classList.add("clicked");
                $panel.classList.add("shown");
            }
        } else if(event.target.closest(".theme_panel .menu_item")){
            let target = event.target.matches(".menu_item") ? event.target : event.target.closest(".menu_item");
            let theme = target.dataset.theme;
            document.querySelectorAll(".theme_panel .menu_item").forEach(div => {
                div.classList.remove("selected");
            });
            postData(URL_FOR_SAVE_THEME,{theme:theme}).then(response => response.text()).then(result => {
                target.classList.add("selected");
                let link = document.createElement("link");
                link.setAttribute("rel","stylesheet");
                link.setAttribute("type","text/css");
                link.setAttribute("href",`/static/css/themes/${theme}.css`);
                document.getElementById('themeDiv').append(link);
            });
        }
});