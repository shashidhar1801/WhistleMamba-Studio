using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Drawing;
using System.Diagnostics;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Web.Script.Serialization;

internal class Configure : Form {
    const string expected="0b3d3b12216dc56935b83d999533df043b850c8f620c08290b692f035458a2e0";
    static readonly string root=AppDomain.CurrentDomain.BaseDirectory;
    static readonly JavaScriptSerializer json=new JavaScriptSerializer();
    readonly TextBox model=new TextBox(), interpreter=new TextBox();
    readonly Label status=new Label();
    readonly Button install=new Button(), choose=new Button(), save=new Button();
    bool busy;
    static string Mounted(string path) {
        string p=Path.GetFullPath(path);
        if(p.Length<3 || p[1]!=':' || p[2]!='\\')throw new Exception("Extract to a local Windows drive. A network or UNC path cannot be used.");
        return "/mnt/"+char.ToLowerInvariant(p[0])+p.Substring(2).Replace('\\','/');
    }
    static string Q(string s) {
        if(s.Contains("\"") || s.Contains("\n") || s.Contains("\r") || s.EndsWith("\\"))throw new Exception("Unsupported character in path.");
        // wsl.exe parses its option prefix separately: leave -d, --cd and
        // --exec unquoted; quote only operands that need Windows quoting.
        return s.Length==0 || s.IndexOfAny(new[]{' ','\t'})>=0 ? "\""+s+"\"" : s;
    }
    static string Call(string work, params string[] argv) {
        var args=new List<string>{"-d","Ubuntu-24.04","--cd",Mounted(Path.Combine(root,work)),"--exec","/usr/bin/env","PYTHONDONTWRITEBYTECODE=1","PYTHONPATH=."};
        args.AddRange(argv);
        var dir=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"WhistleMamba Studio","Logs");
        Directory.CreateDirectory(dir);
        var log=Path.Combine(dir,"configure-"+Guid.NewGuid().ToString("N")+".log");
        var si=new ProcessStartInfo(Path.Combine(Environment.SystemDirectory,"wsl.exe"),string.Join(" ",args.ConvertAll(Q).ToArray()));
        si.UseShellExecute=false;si.CreateNoWindow=true;si.WindowStyle=ProcessWindowStyle.Hidden;
        si.RedirectStandardOutput=si.RedirectStandardError=true;si.RedirectStandardInput=true;
        string receipt=null,last=null;
        using(var w=new StreamWriter(log,false,Encoding.UTF8))using(var p=new Process{StartInfo=si}){
            w.AutoFlush=true;
            DataReceivedEventHandler collect=(s,e)=>{if(e.Data!=null)lock(w){var line=e.Data.Replace("\0","");w.WriteLine(line);
                if(!String.IsNullOrWhiteSpace(line))last=line;
                if(line.StartsWith("STUDIO_CONFIG_OK "))receipt=line.Substring(16);}};
            p.OutputDataReceived+=collect;p.ErrorDataReceived+=collect;p.Start();p.StandardInput.Close();p.BeginOutputReadLine();p.BeginErrorReadLine();p.WaitForExit();
            if(p.ExitCode!=0)throw new Exception((last??"WSL could not complete this step.")+"\r\nLog: "+log);
        }
        return receipt;
    }
    static string Bind(string file,string py) {
        if(String.IsNullOrWhiteSpace(file)||!File.Exists(file))throw new Exception("Choose the separate CSR01 epoch-12 .safetensors file first.");
        file=Path.GetFullPath(file);Mounted(file);
        if(!file.EndsWith(".safetensors",StringComparison.OrdinalIgnoreCase)||new FileInfo(file).Length!=79801968)throw new Exception("Wrong model file type or size. Use the supplied inference-only SafeTensors asset.");
        using(var h=SHA256.Create())using(var f=File.OpenRead(file)){
            if(BitConverter.ToString(h.ComputeHash(f)).Replace("-","").ToLowerInvariant()!=expected)throw new Exception("Model checksum mismatch. No settings were changed.");
        }
        py=py.Trim();
        if(py!=""&&!py.StartsWith("/"))throw new Exception("Use an absolute Linux interpreter path, or leave it blank.");
        var line=Call("app","/usr/bin/python3","-B","check_model.py","locate",py==""?"auto":py,Mounted(file));
        if(line==null)throw new Exception("The GPU did not return a model verification record. Settings were not changed.");
        var d=json.Deserialize<Dictionary<string,string>>(line);
        if(d["weights"]!=expected||!d["python"].StartsWith("/"))throw new Exception("Model identity check failed.");
        var cfg=Path.Combine(root,"LocalRuntime.json");var tmp=cfg+"."+Guid.NewGuid().ToString("N")+".tmp";
        File.WriteAllText(tmp,json.Serialize(new Dictionary<string,string>{{"checkpoint_windows",file},{"distribution","Ubuntu-24.04"},{"linux_python",d["python"]}}),new UTF8Encoding(false));
        try{if(File.Exists(cfg))File.Replace(tmp,cfg,null);else File.Move(tmp,cfg);}finally{if(File.Exists(tmp))File.Delete(tmp);}
        return "Detector ready on "+d["gpu"]+". Open WhistleMamba Browser.exe to analyze the example.";
    }
    static void Install(){Call("runtime","/usr/bin/python3.12","-I","-B","install_runtime.py");}
    void LabelAt(string text,int y){Controls.Add(new Label{Text=text,Left=18,Top=y,AutoSize=true});}
    void ButtonAt(Button b,string text,int x,int y,int width){b.Text=text;b.SetBounds(x,y,width,32);Controls.Add(b);}
    Configure() {
        Text="WhistleMamba Studio | Configure detector";ClientSize=new Size(680,350);Font=new Font("Segoe UI",10);
        StartPosition=FormStartPosition.CenterScreen;FormBorderStyle=FormBorderStyle.FixedDialog;MaximizeBox=false;
        LabelAt("1. On a new PC, install the detector runtime in WSL once (requires Internet).",15);
        ButtonAt(install,"Install / check runtime",18,42,190);
        LabelAt("2. Select the matching CSR01 epoch-12 inference model from a local drive.",88);
        model.SetBounds(18,115,521,28);Controls.Add(model);ButtonAt(choose,"Browse…",550,111,108);
        LabelAt("Existing WSL Python (optional; leave blank for the installed runtime):",158);
        interpreter.SetBounds(18,184,640,27);Controls.Add(interpreter);
        ButtonAt(save,"Verify and save model",18,227,203);status.SetBounds(18,269,640,78);status.Text="Setup checks the model and GPU. It does not open any WAV files.";Controls.Add(status);
        choose.Click+=(s,e)=>{using(var d=new OpenFileDialog{Title="Select the CSR01 inference model",Filter="SafeTensors (*.safetensors)|*.safetensors",CheckFileExists=true})if(d.ShowDialog()==DialogResult.OK)model.Text=d.FileName;};
        install.Click+=async(s,e)=>{if(MessageBox.Show("Install pinned Python and CUDA packages into a separate Ubuntu environment? A first install requires internet, several GB of downloads and disk space. No audio or weights will be uploaded.","Install detector runtime",MessageBoxButtons.OKCancel)!=DialogResult.OK)return;
          await Do("Preparing runtime. On a new PC this may take several minutes…",()=>{Install();return "Runtime checked. Select the model and choose Verify and save model.";});};
        save.Click+=async(s,e)=>{var f=model.Text.Trim();var i=interpreter.Text.Trim();await Do("Checking the model and loading it on the GPU…",()=>Bind(f,i));};
        FormClosing+=(s,e)=>{if(busy)e.Cancel=true;};
    }
    async Task Do(string msg,Func<string> action){busy=true;install.Enabled=choose.Enabled=save.Enabled=false;status.Text=msg;
        try{status.Text=await Task.Run(action);}catch(Exception e){status.Text="Setup did not complete.";MessageBox.Show(e.Message,Text,MessageBoxButtons.OK,MessageBoxIcon.Warning);}
        finally{busy=false;install.Enabled=choose.Enabled=save.Enabled=true;}}
    [STAThread] static int Main(string[] args){
        try{
            if(args.Length==2||args.Length==3){if(args[0]!="--configure")return 2;Console.WriteLine(Bind(args[1],args.Length==3?args[2]:""));return 0;}
            if(args.Length!=0)return 2;
            Application.EnableVisualStyles();Application.SetCompatibleTextRenderingDefault(false);Application.Run(new Configure());return 0;
        }catch(Exception e){Console.Error.WriteLine(e.Message);return 1;}
    }
}
